# Script to create a zip file of your application and upload it to S3 for elastic beanstalk
STACK_NAME=text2sql-app

BUCKET=$1
ZIP_FILE=$2
DBUSER=$3
DBPASSWORD=$4

# create zip 
echo "Creating zip file for your application: $2..."
cd fe
zip -r $ZIP_FILE . -x "*.git*" -x "*.DS_Store*" -x "__pycache__" -x "*.zip" -x "myapp-env*"

# upload to S3
echo "Uploading application zipfile to S3: $1"
apt-get install zip
aws s3 cp $ZIP_FILE s3://$BUCKET/lab-5-text2sql-demo/
cd ..
# create cloudformation stack
aws cloudformation create-stack --stack-name $STACK_NAME --template-body file://template.yaml --parameters ParameterKey=DBUser,ParameterValue=$DBUSER ParameterKey=DBPwd,ParameterValue=$DBPASSWORD
# give elastic beanstalk permission to invoke sagemaker endpoint !!overpermissive, dont use in production!!
aws iam attach-role-policy --role-name aws-elasticbeanstalk-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess