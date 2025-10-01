"""
음성 인식 SubGraph 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.vehicle_agent import VehicleManualAgent
from src.config.settings import DEFAULT_PDF_PATH


def test_speech_recognition():
    """음성 인식 기능 테스트"""
    print("🎤 음성 인식 SubGraph 테스트 시작")
    print("=" * 50)
    
    try:
        # 에이전트 초기화
        print("🔧 에이전트 초기화 중...")
        agent = VehicleManualAgent(str(DEFAULT_PDF_PATH))
        print("✅ 에이전트 초기화 완료!")
        
        # 테스트 1: 텍스트 쿼리 (음성 인식 건너뛰기)
        print("\n📝 테스트 1: 텍스트 쿼리")
        print("-" * 30)
        answer1 = agent.query(user_query="엔진 오일 교체 주기를 알려주세요")
        print(f"답변: {answer1[:100]}...")
        
        # 테스트 2: 음성 인식 (더미)
        print("\n🎤 테스트 2: 음성 인식 (더미)")
        print("-" * 30)
        answer2 = agent.query(audio_data=None, audio_file_path=None)
        print(f"답변: {answer2[:100]}...")
        
        # 테스트 3: 음성 파일 경로 (더미)
        print("\n📁 테스트 3: 음성 파일 경로 (더미)")
        print("-" * 30)
        answer3 = agent.query(audio_file_path="dummy_audio.wav")
        print(f"답변: {answer3[:100]}...")
        
        print("\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()


def test_speech_subgraph_directly():
    """음성 인식 SubGraph 직접 테스트"""
    print("\n🎤 음성 인식 SubGraph 직접 테스트")
    print("=" * 50)
    
    try:
        from src.agents.subgraphs.speech_recognition_subgraph import SpeechRecognitionSubGraph
        
        # SubGraph 초기화
        speech_subgraph = SpeechRecognitionSubGraph()
        
        # 테스트 1: 더미 음성 데이터
        print("📊 테스트 1: 더미 음성 데이터")
        result1 = speech_subgraph.invoke(audio_data=None, audio_file_path=None)
        print(f"인식 결과: {result1['final_text']}")
        print(f"신뢰도: {result1['confidence']:.2f}")
        print(f"오류: {result1['error']}")
        
        # 테스트 2: 더미 음성 파일
        print("\n📁 테스트 2: 더미 음성 파일")
        result2 = speech_subgraph.invoke(audio_file_path="test_audio.wav")
        print(f"인식 결과: {result2['final_text']}")
        print(f"신뢰도: {result2['confidence']:.2f}")
        print(f"오류: {result2['error']}")
        
        print("\n✅ SubGraph 직접 테스트 완료!")
        
    except Exception as e:
        print(f"❌ SubGraph 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚗 차량 매뉴얼 RAG - 음성 인식 테스트")
    print("=" * 60)
    
    # 전체 에이전트 테스트
    test_speech_recognition()
    
    # SubGraph 직접 테스트
    test_speech_subgraph_directly()
    
    print("\n🎉 모든 테스트 완료!")
