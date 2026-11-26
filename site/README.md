# site — stardust 영문 사이트

stardust의 영문 홈페이지(Astro). 옛 한글 사이트는 `site-legacy/`에 참고용으로 남겨 두었다(배포 안 함). 이 폴더만 이 사이트 세션이 고치고, 다른 세션은 건드리지 않는다.

## 기준 빌드
- 커밋: `fa8db8404e84d3f61e980b130312500472341171` (2026-11-27 00:48 +0900). `sd.cluster`는 CL1(계층 군집)까지만 있어 문서에서 뺀다.
- 사용자가 버전업을 말하기 전까지 이 커밋에 고정한다. 작업 중인 저장소의 소스를 보지 않고, 예제 실행·시그니처 대조는 모두 이 빌드로 한다.
- 복제본: `site/.sdbuild/stardust` (git 무시)
- 인터프리터: `site/.sdbuild/stardust/.venv/bin/python` (Python 3.15, `[test]` 묶음 포함). `sd.__file__`이 `.sdbuild/stardust/src/stardust/__init__.py`이면 맞다.
- 만든 방법:
  ```bash
  git clone /Users/sanghyunpark/Desktop/stardust site/.sdbuild/stardust
  cd site/.sdbuild/stardust && git checkout fa8db8404e84d3f61e980b130312500472341171
  python3.15 -m venv .venv && .venv/bin/pip install nanobind scikit-build-core
  CMAKE_BUILD_PARALLEL_LEVEL=8 .venv/bin/pip install --no-build-isolation -e ".[test]"
  ```
- 버전업할 때: 복제본에서 `git fetch origin && git checkout <커밋>` 뒤 마지막 줄을 다시 돌리고, 이 절과 `API-INVENTORY.md`(`tools/api_inventory.py`)를 다시 만든다.

## API 목록
- `.sdbuild/stardust/.venv/bin/python -I tools/api_inventory.py` → `API-INVENTORY.md`(커밋), `notes/api-inventory.json`(독스트링·클래스 멤버 전체, 커밋 안 함).
- 기준 빌드의 docstring은 대부분 시그니처 한 줄뿐이다. 함수의 뜻은 복제본의 `API.md`, `DESIGN.md`, `docs/`, 소스와 실제 실행으로 확인한다.

## 사이트
- `pnpm install`, `pnpm dev`, `pnpm build`. 디자인(토큰·컴포넌트·히어로)은 옛 사이트(`site-legacy/`)에서 가져왔고, 본문은 새로 쓴다.
- 히어로는 충돌 장면 없이 완성된 은하(`src/scripts/stardust.ts`)만 돈다.
- 작업 노트는 `notes/`(git 무시).
