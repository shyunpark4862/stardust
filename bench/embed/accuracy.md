# sd.embed 정확도

스윕이 생기면 `.venv/bin/python -m bench.accuracy promote`가 그 기준선에서 만든다. 손으로 고치지 않는다.

아직 스윕이 없다. 참값(tests/embed/ref, tools/embed_ref.py)과 비교 대상의 기록(tests/embed/data)은 EM0 마일스톤에서 만들었고, 스윕은 공개 함수가 생기는 EM1–EM8 마일스톤에서 bench/accuracy/embed.py에 더한다.
