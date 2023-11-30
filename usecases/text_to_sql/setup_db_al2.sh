sudo yum install -y postgresql postgresql-server # install postgresql client and server
sudo postgresql-setup initdb # initialize the db
# enable and start the service
sudo systemctl enable postgresql.service
sudo systemctl start postgresql.service
# create user and db
sudo -u postgres createuser "ec2-user"
sudo -u postgres createdb "companydb"
# populate the db
psql -d companydb -f "fe/populate.sql"