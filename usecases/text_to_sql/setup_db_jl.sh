sudo apt update -y
sudo apt install -y postgresql postgresql-contrib
sudo service postgresql start
cd /
sudo -u postgres createuser "sagemaker-user"
sudo -u postgres createdb "companydb"
cd "$OLDPWD"
psql -d companydb -f "fe/populate.sql"