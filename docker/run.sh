cd docker
export GID=$(id -g)
export GROUPNAME=$(echo $(id -Gn) | awk '{print $1}')
docker compose up -d --build
docker compose exec smp_ws bash
docker compose down
cd ..
