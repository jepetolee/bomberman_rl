# Configuration Files

## trm_config.yaml

TRM (Tiny Recursive Model) 학습을 위한 모든 하이퍼파라미터와 설정을 포함하는 YAML 파일입니다.

### 사용 방법

#### 1. 기본 설정 로드

```python
from config.load_config import load_config, apply_config_to_env

# 설정 로드
config = load_config('config/trm_config.yaml')

# 환경변수에 적용
apply_config_to_env(config)
```

#### 2. Preset 사용

```python
from config.load_config import load_config, apply_preset, apply_config_to_env

# 기본 설정 로드
config = load_config()

# Preset 적용 (예: small, medium, large)
config = apply_preset(config, 'small')

# 환경변수에 적용
apply_config_to_env(config)
```

#### 3. 명령줄에서 사용

```bash
# 설정 로드 및 환경변수 적용
python config/load_config.py --apply-env

# Preset 적용
python config/load_config.py --preset small --apply-env

# 설정 확인
python config/load_config.py --print
```

#### 4. 스크립트에서 사용

```bash
# 환경변수로 적용
source <(python config/load_config.py --preset medium --apply-env)

# 또는 export
eval $(python config/load_config.py --preset medium --apply-env --export)
```

### Presets

- **small**: 빠른 학습용 작은 모델
- **medium**: 기본 설정 (default)
- **large**: 더 나은 성능을 위한 큰 모델
- **overlapping_patches**: Overlapping patches로 더 많은 정보 제공
- **high_supervision**: 더 많은 supervision steps

### 주요 설정 섹션

1. **model**: 모델 아키텍처 설정 (embed_dim, z_dim, n_latent, etc.)
2. **phase1**: Teacher 데이터 수집 설정
3. **phase2**: Dyna-Q Planning + DeepSupervision 설정
4. **phase3**: Recurrent RL 학습 설정
5. **training**: 일반 학습 설정
6. **paths**: 경로 설정
7. **env_vars**: 환경변수 매핑

### 환경변수 자동 설정

YAML 파일의 `env_vars` 섹션에 정의된 값들이 자동으로 환경변수로 설정됩니다:

```yaml
env_vars:
  BOMBER_USE_TRM: "1"
  BOMBER_TRM_N: "6"
  BOMBER_TRM_N_SUP: "16"
  # ... etc
```

이렇게 설정된 환경변수는 모든 학습 스크립트에서 자동으로 사용됩니다.

