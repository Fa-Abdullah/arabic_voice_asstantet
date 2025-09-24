"""
Fixed Arabic Voice Assistant - Streamlit Interface
With working voice recording and proper Arabic responses
"""

import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
import base64
from pathlib import Path
import json
import time
import threading

# Core imports
try:
    from vosk import Model, KaldiRecognizer
    from gtts import gTTS
    from openai import OpenAI
except ImportError as e:
    st.error(f"Missing package: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🦷 عيادة فانكوفر لطب الأسنان",
    page_icon="🦷",
    layout="wide"
)

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'assistant_ready' not in st.session_state:
    st.session_state.assistant_ready = False
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'chat_mode' not in st.session_state:
    st.session_state.chat_mode = "written"  # "written", "audio", or "twins"
if 'auto_record' not in st.session_state:
    st.session_state.auto_record = False
if 'first_message_sent' not in st.session_state:
    st.session_state.first_message_sent = False

class ArabicVoiceAssistant:
    def __init__(self):
        """Initialize Arabic Voice Assistant"""
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
        )
        
        self.model = None
        self.rec = None
        self.sample_rate = 16000
        
        # Pure Arabic system prompt
        self.system_prompt = """أنت ساندي، موظفة استقبال في عيادة فانكوفر لطب الأسنان.

المعلومات المهمة:
- أوقات العمل: الاثنين إلى الجمعة من 8 صباحاً إلى 6 مساءً، السبت من 9 صباحاً إلى 3 مساءً
- الموقع: وسط مدينة فانكوفر
- الهاتف: (604) 555-DENTAL
- الخدمات: طب الأسنان العام، تنظيف الأسنان، الحشوات، التيجان، علاج الجذور، طب الأسنان التجميلي

تعليمات مهمة:
1. اجيبي باللغة العربية فقط
2. كوني ودودة ومهنية
3. اجعلي الردود قصيرة وواضحة
4. اسألي دائماً كيف يمكنك المساعدة أكثر
5. عند حجز المواعيد، اطلبي اسم المريض ونوع الخدمة المطلوبة

مثال للترحيب: "مرحباً، أهلاً وسهلاً بك في عيادة فانكوفر لطب الأسنان. اسمي ساندي، كيف يمكنني مساعدتك اليوم؟"
"""

    def find_and_load_model(self):
        """Find and load Vosk Arabic model"""
        try:
            search_paths = [Path.cwd(), Path.home() / "Downloads"]
            
            for search_dir in search_paths:
                if not search_dir.exists():
                    continue
                    
                for item in search_dir.iterdir():
                    if (item.is_dir() and 
                        'vosk' in item.name.lower() and 
                        'ar' in item.name.lower()):
                        
                        if (item / 'am').exists() and (item / 'graph').exists():
                            self.model = Model(str(item))
                            self.rec = KaldiRecognizer(self.model, self.sample_rate)
                            return True, f"✅ تم تحميل النموذج من: {item.name}"
            
            return False, "❌ لم يتم العثور على نموذج Vosk العربي"
        
        except Exception as e:
            return False, f"❌ خطأ في تحميل النموذج: {e}"

    def transcribe_audio_file(self, audio_file_path):
        """Transcribe audio file using Vosk"""
        if not self.rec:
            return ""
        
        try:
            # Reset recognizer for new audio
            self.rec = KaldiRecognizer(self.model, self.sample_rate)
            
            # Read WAV file
            with wave.open(audio_file_path, 'rb') as wav_file:
                # Check audio format
                if wav_file.getframerate() != self.sample_rate:
                    st.warning(f"⚠️ معدل العينة غير متطابق: {wav_file.getframerate()} Hz")
                
                # Read audio data
                audio_data = wav_file.readframes(wav_file.getnframes())
                
                # Process with Vosk
                if self.rec.AcceptWaveform(audio_data):
                    result = json.loads(self.rec.Result())
                    text = result.get("text", "").strip()
                else:
                    # Get final result
                    result = json.loads(self.rec.FinalResult())
                    text = result.get("text", "").strip()
                
                return text
        
        except Exception as e:
            st.error(f"خطأ في تحويل الصوت إلى نص: {e}")
            return ""

    def generate_response(self, user_text):
        """Generate Arabic response"""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add recent chat history (last 5 messages)
            for msg in st.session_state.chat_history[-5:]:
                messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["assistant"]})
            
            # Add current user message
            messages.append({"role": "user", "content": user_text})
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="google/gemma-2-9b-it",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return "عذراً، حدث خطأ تقني. يرجى المحاولة مرة أخرى."

    def generate_greeting(self):
        """Generate a greeting message"""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.append({"role": "user", "content": "مرحباً"})
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="google/gemma-2-9b-it",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return "مرحباً، أهلاً وسهلاً بك في عيادة فانكوفر لطب الأسنان. اسمي ساندي، كيف يمكنني مساعدتك اليوم؟"

    def text_to_speech(self, text):
        """Generate Arabic TTS"""
        try:
            tts = gTTS(text=text, lang='ar', slow=False)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)
            return temp_file.name
        except Exception as e:
            st.error(f"خطأ في تحويل النص إلى صوت: {e}")
            return None

# Initialize assistant
@st.cache_resource
def get_assistant():
    return ArabicVoiceAssistant()

def record_audio(duration=5):
    """Record audio for specified duration"""
    try:
        st.info(f"🔴 جاري التسجيل... ({duration} ثواني)")
        
        # Record audio
        audio_data = sd.rec(int(duration * 16000), 
                          samplerate=16000, 
                          channels=1, 
                          dtype='float32')
        
        # Show countdown
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(duration):
            progress_bar.progress((i + 1) / duration)
            status_text.text(f"🎤 التسجيل جاري... {duration - i} ثواني متبقية")
            time.sleep(1)
        
        sd.wait()
        progress_bar.empty()
        status_text.empty()
        
        # Convert to int16 and save as WAV
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_int16.tobytes())
        
        return temp_file.name
    
    except Exception as e:
        st.error(f"خطأ في التسجيل: {e}")
        return None

def play_audio(audio_file):
    """Play audio in browser"""
    if os.path.exists(audio_file):
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        audio_html = f"""
        <audio controls autoplay style="width: 100%;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            المتصفح لا يدعم تشغيل الصوت
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

def send_greeting(assistant):
    """Send greeting message from assistant"""
    if not st.session_state.first_message_sent:
        with st.spinner("🤖 جاري التحضير..."):
            greeting = assistant.generate_greeting()
            
            # Add to chat history
            st.session_state.chat_history.append({
                "user": "",
                "assistant": greeting
            })
            
            # Generate TTS for audio and twins modes
            if st.session_state.chat_mode in ["audio", "twins"]:
                with st.spinner("🔊 جاري تحويل التحية إلى صوت..."):
                    audio_file = assistant.text_to_speech(greeting)
                
                if audio_file:
                    # Store audio reference
                    st.session_state["audio_greeting"] = audio_file
                    
                    # Play audio
                    play_audio(audio_file)
        
        st.session_state.first_message_sent = True
        st.rerun()

def main():
    """Main application"""
    
    # Header
    st.title("🦷 عيادة فانكوفر لطب الأسنان")
    st.markdown("### 🤖 مساعدة الاستقبال الذكية - ساندي")
    st.markdown("**مرحباً وأهلاً بكم! يمكنكم التحدث باللغة العربية**")
    
    # Initialize assistant
    assistant = get_assistant()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ إعدادات النظام")
        
        # Load model
        if not st.session_state.assistant_ready:
            with st.spinner("🔄 تحميل النموذج..."):
                success, message = assistant.find_and_load_model()
                st.info(message)
                
                if success:
                    st.session_state.assistant_ready = True
                else:
                    st.error("تأكد من وجود نموذج Vosk العربي")
                    st.stop()
        else:
            st.success("✅ النظام جاهز")
        
        st.divider()
        
        # Chat mode selection
        st.header("💬 نمط المحادثة")
        chat_mode = st.radio(
            "اختر نمط المحادثة:",
            ["نصي", "صوتي", "توأم"],
            captions=["محادثة نصية عادية", "تسجيل صوتي تلقائي", "نص وصوت معاً"],
            index=0
        )
        
        # Map selection to mode
        mode_map = {"نصي": "written", "صوتي": "audio", "توأم": "twins"}
        st.session_state.chat_mode = mode_map[chat_mode]
        
        # Recording settings for audio modes
        if st.session_state.chat_mode in ["audio", "twins"]:
            st.subheader("🎤 إعدادات التسجيل")
            record_duration = st.slider("مدة التسجيل (ثانية)", 3, 10, 5)
            
            if st.session_state.chat_mode == "audio":
                st.session_state.auto_record = st.checkbox("التسجيل التلقائي بعد الرد", value=True)
        
        st.divider()
        
        # Clear chat
        if st.button("🗑️ مسح المحادثة"):
            st.session_state.chat_history = []
            st.session_state.first_message_sent = False
            st.rerun()
        
        st.divider()
        
        # Clinic info
        st.header("🏥 معلومات العيادة")
        st.info("""
        **عيادة فانكوفر لطب الأسنان**
        
        📍 وسط مدينة فانكوفر  
        📞 (604) 555-DENTAL
        
        **أوقات العمل:**
        • الاثنين-الجمعة: 8ص-6م
        • السبت: 9ص-3م  
        • الأحد: مغلق
        
        **الخدمات:**
        • طب الأسنان العام
        • تنظيف وفحص الأسنان
        • الحشوات والتيجان
        • علاج الجذور
        • طب الأسنان التجميلي
        """)

    # Send greeting message if not already sent
    if st.session_state.assistant_ready and not st.session_state.first_message_sent:
        send_greeting(assistant)

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 المحادثة")
        
        # Display current mode
        mode_display = {
            "written": "📝 النمط النصي",
            "audio": "🎤 النمط الصوتي",
            "twins": "👥 النمط التوأم (نصي وصوتي)"
        }
        st.info(f"**النمط الحالي:** {mode_display[st.session_state.chat_mode]}")
        
        # Voice input section for audio and twins modes
        if st.session_state.chat_mode in ["audio", "twins"]:
            st.subheader("🎤 الإدخال الصوتي")
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                if st.button("🎙️ ابدأ التسجيل", disabled=not st.session_state.assistant_ready):
                    audio_file = record_audio(record_duration)
                    if audio_file:
                        st.session_state.last_recording = audio_file
                        st.success("✅ تم التسجيل بنجاح!")
                        
                        # Play recorded audio
                        st.audio(audio_file, format="audio/wav")
            
            with col_rec2:
                if st.button("🎯 معالجة التسجيل") and hasattr(st.session_state, 'last_recording'):
                    with st.spinner("🔄 جاري تحويل الصوت إلى نص..."):
                        text = assistant.transcribe_audio_file(st.session_state.last_recording)
                        
                        if text:
                            process_user_input(assistant, text)
                        else:
                            st.warning("⚠️ لم يتم التعرف على كلام. حاول مرة أخرى.")
            
            # File upload section
            st.subheader("📁 رفع ملف صوتي")
            uploaded_file = st.file_uploader("اختر ملف صوتي (WAV)", type=['wav'], key="audio_uploader")
            
            if uploaded_file and st.button("🎯 معالجة الملف المرفوع", key="process_uploaded"):
                # Save uploaded file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_file.write(uploaded_file.read())
                temp_file.close()
                
                with st.spinner("🔄 معالجة الملف الصوتي..."):
                    text = assistant.transcribe_audio_file(temp_file.name)
                    
                    if text:
                        process_user_input(assistant, text)
                    else:
                        st.warning("⚠️ لم يتم التعرف على كلام في الملف.")
                
                os.unlink(temp_file.name)
        
        # Text input for written and twins modes
        if st.session_state.chat_mode in ["written", "twins"]:
            st.subheader("⌨️ الإدخال النصي")
            user_text = st.text_area("اكتب رسالتك هنا:", height=100, 
                                    placeholder="مثال: ما هي مواعيد العيادة؟", key="text_input")
            
            if st.button("📤 إرسال الرسالة", key="send_text") and user_text.strip():
                process_user_input(assistant, user_text.strip())
        
        # Chat history
        st.divider()
        st.subheader("📋 سجل المحادثة")
        
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["user"]:  # Only show user messages if they exist
                st.markdown(f"**👤 المريض:** {msg['user']}")
            st.markdown(f"**🤖 ساندي:** {msg['assistant']}")
            
            # Audio playback for audio and twins modes
            if st.session_state.chat_mode in ["audio", "twins"]:
                audio_key = f"audio_{i}" if i > 0 else "audio_greeting"
                if audio_key in st.session_state:
                    audio_file = st.session_state[audio_key]
                    if os.path.exists(audio_file):
                        st.audio(audio_file, format="audio/mp3")
            
            st.divider()

    with col2:
        st.header("📊 حالة النظام")
        
        st.metric("🤖 نموذج الذكاء الاصطناعي", "جاهز" if st.session_state.assistant_ready else "تحميل")
        st.metric("💬 عدد الرسائل", len(st.session_state.chat_history))
        st.metric("🎯 نمط المحادثة", list(mode_display.values())[list(mode_display.keys()).index(st.session_state.chat_mode)])
        
        if st.session_state.chat_history:
            last_msg = st.session_state.chat_history[-1]
            if last_msg["user"]:
                st.text_area("آخر رسالة:", value=last_msg['user'], height=100, disabled=True)
            else:
                st.text_area("آخر رسالة:", value=last_msg['assistant'], height=100, disabled=True)
        
        # Microphone test
        st.divider()
        st.subheader("🎤 اختبار الميكروفون")
        if st.button("اختبار الميكروفون"):
            test_microphone()

def process_user_input(assistant, text):
    """Process user input and generate response"""
    st.success(f"🎯 **النص المستخرج:** {text}")
    
    # Generate response
    with st.spinner("🤖 جاري إنشاء الرد..."):
        response = assistant.generate_response(text)
    
    # Add to chat history
    st.session_state.chat_history.append({
        "user": text,
        "assistant": response
    })
    
    # Generate TTS for audio and twins modes
    if st.session_state.chat_mode in ["audio", "twins"]:
        with st.spinner("🔊 جاري تحويل الرد إلى صوت..."):
            audio_file = assistant.text_to_speech(response)
        
        if audio_file:
            # Store audio reference
            msg_index = len(st.session_state.chat_history) - 1
            st.session_state[f"audio_{msg_index}"] = audio_file
            
            # Play audio
            st.success("🔊 تشغيل الرد...")
            time.sleep(8)  # Ensure TTS file is ready
            play_audio(audio_file)
            
            # Auto-record for audio mode
            if st.session_state.chat_mode == "audio" and st.session_state.auto_record:
                st.info("⏳ الانتظار قليلاً قبل التسجيل التلقائي...")
                time.sleep(3)  # Wait before auto-recording
                st.rerun()
    
    st.rerun()

def test_microphone():
    """Test microphone"""
    try:
        st.info("🎤 اختبار الميكروفون لمدة 3 ثوان...")
        
        audio_data = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
        
        progress_bar = st.progress(0)
        for i in range(3):
            progress_bar.progress((i + 1) / 3)
            time.sleep(1)
        
        sd.wait()
        progress_bar.empty()
        
        volume = np.abs(audio_data).mean()
        
        if volume > 0.001:
            st.success(f"✅ الميكروفون يعمل بشكل جيد! مستوى الصوت: {volume:.4f}")
        else:
            st.error("❌ لم يتم اكتشاف صوت. تحقق من إعدادات الميكروفون.")
            
    except Exception as e:
        st.error(f"❌ فشل اختبار الميكروفون: {e}")

if __name__ == "__main__":
    main()


#     # 1️⃣ اسحبي الملفات اللي على GitHub وادمجيها حتى لو مش مرتبطة
# git pull origin main --allow-unrelated-histories

# # 2️⃣ لو حصل تعارضات (conflicts)، افتحي الملفات اللي فيها conflict وعدليها، 
# #    بعدين أضيفي التعديلات:
# git add .

# # 3️⃣ اعملي commit بعد حل التعارضات
# git commit -m "دمج التغييرات مع الريبو الأصلي"

# # 4️⃣ ارفعي من جديد
# git push -u origin main
