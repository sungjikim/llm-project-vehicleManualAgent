"""
음성 인식 SubGraph
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
import io
import wave
import tempfile
import os

from ...models.subgraph_states import SpeechRecognitionState


class DummyASR:
    """더미 ASR (Automatic Speech Recognition) 클래스"""
    
    def __init__(self):
        self.model_name = "dummy-asr-model"
        self.sample_rate = 16000
        self.chunk_size = 1024
    
    def transcribe(self, audio_data: bytes) -> str:
        """
        음성 데이터를 텍스트로 변환 (더미 구현)
        
        Args:
            audio_data: 음성 오디오 데이터 (bytes)
            
        Returns:
            str: 변환된 텍스트
        """
        # 더미 구현: 실제로는 Whisper, Google Speech-to-Text 등을 사용
        print("🎤 [DummyASR] 음성 인식 중...")
        
        # 실제 구현에서는 다음과 같이 사용:
        # import whisper
        # model = whisper.load_model("base")
        # result = model.transcribe(audio_file_path)
        # return result["text"]
        
        # 더미 응답
        dummy_responses = [
            "엔진 오일 교체 주기를 알려주세요",
            "브레이크에서 소리가 나는데 어떻게 해야 하나요",
            "타이어 공기압은 얼마로 맞춰야 하나요",
            "지금 운전 중인데 경고등이 켜졌어요",
            "겨울철 차량 관리 방법을 알려주세요"
        ]
        
        import random
        dummy_text = random.choice(dummy_responses)
        print(f"🎤 [DummyASR] 인식된 텍스트: {dummy_text}")
        
        return dummy_text
    
    def is_audio_valid(self, audio_data: bytes) -> bool:
        """
        음성 데이터 유효성 검사
        
        Args:
            audio_data: 음성 오디오 데이터
            
        Returns:
            bool: 유효한 음성 데이터인지 여부
        """
        try:
            # WAV 파일 헤더 검사
            if len(audio_data) < 44:  # WAV 파일 최소 크기
                return False
            
            # RIFF 헤더 확인
            if audio_data[:4] != b'RIFF':
                return False
            
            # WAVE 포맷 확인
            if audio_data[8:12] != b'WAVE':
                return False
            
            return True
        except Exception:
            return False


class DummySTT:
    """더미 STT (Speech-to-Text) 클래스"""
    
    def __init__(self):
        self.language = "ko-KR"  # 한국어
        self.encoding = "LINEAR16"
        self.sample_rate = 16000
    
    def process_audio(self, audio_file_path: str) -> str:
        """
        음성 파일을 텍스트로 변환 (더미 구현)
        
        Args:
            audio_file_path: 음성 파일 경로
            
        Returns:
            str: 변환된 텍스트
        """
        print(f"🎵 [DummySTT] 음성 파일 처리 중: {audio_file_path}")
        
        try:
            # 실제 구현에서는 다음과 같이 사용:
            # import speech_recognition as sr
            # r = sr.Recognizer()
            # with sr.AudioFile(audio_file_path) as source:
            #     audio = r.record(source)
            #     text = r.recognize_google(audio, language='ko-KR')
            #     return text
            
            # 더미 구현: 파일 크기 기반으로 다른 응답 반환
            file_size = os.path.getsize(audio_file_path)
            
            if file_size < 1000:  # 작은 파일
                return "엔진 오일 교체 주기를 알려주세요"
            elif file_size < 5000:  # 중간 파일
                return "브레이크에서 소리가 나는데 어떻게 해야 하나요"
            else:  # 큰 파일
                return "지금 운전 중인데 경고등이 켜졌어요"
                
        except Exception as e:
            print(f"❌ [DummySTT] 음성 처리 오류: {str(e)}")
            return "음성 인식에 실패했습니다. 다시 말씀해 주세요."
    
    def create_dummy_audio_file(self, duration: float = 2.0) -> str:
        """
        더미 음성 파일 생성 (테스트용)
        
        Args:
            duration: 음성 길이 (초)
            
        Returns:
            str: 생성된 음성 파일 경로
        """
        # 임시 WAV 파일 생성
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file_path = temp_file.name
        temp_file.close()
        
        # 더미 WAV 파일 생성 (무음)
        sample_rate = 16000
        num_samples = int(sample_rate * duration)
        
        with wave.open(temp_file_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # 모노
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # 무음 데이터 생성
            silence = b'\x00' * (num_samples * 2)
            wav_file.writeframes(silence)
        
        return temp_file_path


class SpeechRecognitionSubGraph:
    """음성 인식 SubGraph"""
    
    def __init__(self):
        self.asr = DummyASR()
        self.stt = DummySTT()
    
    def audio_processor(self, state: SpeechRecognitionState) -> Dict[str, Any]:
        """음성 데이터 처리 노드"""
        audio_data = state.get("audio_data")
        audio_file_path = state.get("audio_file_path")
        
        try:
            print("🎤 음성 인식 SubGraph 실행 중...")
            
            if audio_data:
                # 바이트 데이터로 직접 처리
                print("📊 바이트 데이터로 음성 인식 중...")
                
                if not self.asr.is_audio_valid(audio_data):
                    return {
                        "recognized_text": "",
                        "confidence": 0.0,
                        "error": "유효하지 않은 음성 데이터입니다.",
                        "processing_method": "asr_bytes"
                    }
                
                recognized_text = self.asr.transcribe(audio_data)
                confidence = 0.85  # 더미 신뢰도
                
            elif audio_file_path:
                # 파일 경로로 처리
                print(f"📁 파일 경로로 음성 인식 중: {audio_file_path}")
                
                if not os.path.exists(audio_file_path):
                    return {
                        "recognized_text": "",
                        "confidence": 0.0,
                        "error": "음성 파일을 찾을 수 없습니다.",
                        "processing_method": "stt_file"
                    }
                
                recognized_text = self.stt.process_audio(audio_file_path)
                confidence = 0.80  # 더미 신뢰도
                
            else:
                # 더미 음성 파일 생성 (테스트용)
                print("🎵 더미 음성 파일 생성 중...")
                dummy_file = self.stt.create_dummy_audio_file()
                recognized_text = self.stt.process_audio(dummy_file)
                confidence = 0.75
                
                # 임시 파일 정리
                try:
                    os.unlink(dummy_file)
                except:
                    pass
            
            print(f"✅ 음성 인식 완료: '{recognized_text}' (신뢰도: {confidence:.2f})")
            
            return {
                "recognized_text": recognized_text,
                "confidence": confidence,
                "error": None,
                "processing_method": "asr_bytes" if audio_data else "stt_file"
            }
            
        except Exception as e:
            print(f"❌ 음성 인식 오류: {str(e)}")
            return {
                "recognized_text": "",
                "confidence": 0.0,
                "error": f"음성 인식 중 오류가 발생했습니다: {str(e)}",
                "processing_method": "error"
            }
    
    def text_validator(self, state: SpeechRecognitionState) -> Dict[str, Any]:
        """인식된 텍스트 검증 노드"""
        recognized_text = state.get("recognized_text", "")
        confidence = state.get("confidence", 0.0)
        
        try:
            print("🔍 인식된 텍스트 검증 중...")
            
            # 빈 텍스트 검사
            if not recognized_text or recognized_text.strip() == "":
                return {
                    "is_valid": False,
                    "validation_error": "인식된 텍스트가 없습니다.",
                    "final_text": ""
                }
            
            # 신뢰도 검사
            if confidence < 0.3:
                return {
                    "is_valid": False,
                    "validation_error": f"음성 인식 신뢰도가 너무 낮습니다 ({confidence:.2f})",
                    "final_text": recognized_text
                }
            
            # 텍스트 길이 검사
            if len(recognized_text) < 2:
                return {
                    "is_valid": False,
                    "validation_error": "인식된 텍스트가 너무 짧습니다.",
                    "final_text": recognized_text
                }
            
            # 텍스트 정리
            cleaned_text = recognized_text.strip()
            
            print(f"✅ 텍스트 검증 완료: '{cleaned_text}'")
            
            return {
                "is_valid": True,
                "validation_error": None,
                "final_text": cleaned_text
            }
            
        except Exception as e:
            print(f"❌ 텍스트 검증 오류: {str(e)}")
            return {
                "is_valid": False,
                "validation_error": f"텍스트 검증 중 오류가 발생했습니다: {str(e)}",
                "final_text": recognized_text
            }
    
    def create_graph(self) -> StateGraph:
        """음성 인식 SubGraph 생성"""
        workflow = StateGraph(SpeechRecognitionState)
        
        # 노드 추가
        workflow.add_node("audio_processor", self.audio_processor)
        workflow.add_node("text_validator", self.text_validator)
        
        # 엣지 추가
        workflow.set_entry_point("audio_processor")
        workflow.add_edge("audio_processor", "text_validator")
        workflow.add_edge("text_validator", END)
        
        return workflow.compile()
    
    def invoke(self, audio_data: Optional[bytes] = None, 
               audio_file_path: Optional[str] = None) -> Dict[str, Any]:
        """SubGraph 실행"""
        graph = self.create_graph()
        
        initial_state = {
            "audio_data": audio_data,
            "audio_file_path": audio_file_path,
            "recognized_text": "",
            "confidence": 0.0,
            "error": None,
            "processing_method": "",
            "is_valid": False,
            "validation_error": None,
            "final_text": ""
        }
        
        return graph.invoke(initial_state)
