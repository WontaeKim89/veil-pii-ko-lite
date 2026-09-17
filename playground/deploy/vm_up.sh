#!/usr/bin/env bash
# Azure VM 1대(CPU) 생성 → docker 설치 → 레포 동기화 → .env/basic auth 준비 → compose up. 멱등(이미 있으면 건너뜀).
# 사용: SUB=<subscription id> RG=<rg> [VM=vm-pii-playground SIZE=Standard_D8s_v5 LOCATION=koreacentral] \
#       AZ_LANG_ENDPOINT=... AZ_LANG_KEY=... DEMO_PASS=... bash playground/deploy/vm_up.sh
set -euo pipefail
SUB=${SUB:?subscription id}; RG=${RG:?resource group}; LOCATION=${LOCATION:-koreacentral}
VM=${VM:-vm-pii-playground}; SIZE=${SIZE:-Standard_D8s_v5}; ADMIN=${ADMIN:-azureuser}; KEY=${KEY:-$HOME/.ssh/id_ed25519_gpu_vm}
DEMO_PASS=${DEMO_PASS:?basic auth password}
: "${AZ_LANG_ENDPOINT:?}" "${AZ_LANG_KEY:?}"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
azs(){ az "$@" --subscription "$SUB"; }

if ! azs vm show -g "$RG" -n "$VM" -o none 2>/dev/null; then
  azs vm create -g "$RG" -n "$VM" -l "$LOCATION" --image Ubuntu2404 --size "$SIZE" --os-disk-size-gb 128 \
    --admin-username "$ADMIN" --ssh-key-values "$KEY.pub" --public-ip-sku Standard --nsg-rule SSH -o none
  azs vm open-port -g "$RG" -n "$VM" --port 80,443 --priority 1010 -o none
fi
IP=$(azs vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv); echo "VM $VM → $IP"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 $ADMIN@$IP"
for i in $(seq 1 20); do $SSH true 2>/dev/null && break; sleep 10; done

$SSH 'command -v docker >/dev/null || { curl -fsSL https://get.docker.com | sudo sh; }; sudo usermod -aG docker $USER'
(cd "$ROOT" && rsync -azR --exclude '__pycache__' --exclude 'release/fp32' --exclude '*.int4.onnx' \
   -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" release eval train configs playground .dockerignore "$ADMIN@$IP:~/kopii-lite/")

# 비밀은 VM 의 .env 로만 (stdin 경유, 로그에 안 남음)
printf 'AZ_LANG_ENDPOINT=%s\nAZ_LANG_KEY=%s\nAZURE_RPM=10\n' "$AZ_LANG_ENDPOINT" "$AZ_LANG_KEY" | $SSH 'umask 077; cat > ~/kopii-lite/playground/.env'
HASH=$($SSH "sudo docker run --rm caddy:2 caddy hash-password --plaintext '$DEMO_PASS'")
$SSH "sed -i 's#demo \\\$2a\\\$14\\\$REPLACE_WITH_BCRYPT_HASH#demo $HASH#' ~/kopii-lite/playground/Caddyfile"
$SSH 'cd ~/kopii-lite/playground && sudo docker compose up -d --build && sudo docker compose ps'
echo "→ https://$IP  (basic auth demo / \$DEMO_PASS)"
