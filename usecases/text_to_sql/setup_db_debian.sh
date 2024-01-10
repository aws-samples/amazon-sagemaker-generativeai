sh -c 'echo "deb https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list' >/dev/null 2>&1
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - >/dev/null 2>&1
apt update >/dev/null 2>&1
apt-get -y install gnupg >/dev/null 2>&1
apt install -y postgresql-common >/dev/null 2>&1
/usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y >/dev/null 2>&1
service postgresql start >/dev/null 2>&1
su postgres -c "createuser root" >/dev/null 2>&1
su postgres -c "createdb companydb" >/dev/null 2>&1
psql -d companydb -f "fe/populate.sql" >/dev/null 2>&1