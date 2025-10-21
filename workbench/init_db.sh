docker exec -i $(sudo docker ps | grep "postgres\|postgres" | cut -f1 -d" ") \
psql -U postgres << EOF
CREATE DATABASE "starfish-router";
EOF
