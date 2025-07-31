import requests
import boto3


# Get the access token from Cognito
def get_token(gateway_id: str, region_name: str = None, client_id: str = None) -> dict:

    region_name = region_name if region_name else boto3.Session().region_name
    # scope_string = f"{gateway_id}/gateway:read {gateway_id}/gateway:write"

    agentcore_client = boto3.client('bedrock-agentcore-control')
    cognito_client = boto3.client("cognito-idp", region_name=region_name)

    try:
        # Get token endpoint from agentcore gateway
        gateway = agentcore_client.get_gateway(gatewayIdentifier=gateway_id)
        discovery_url = gateway["authorizerConfiguration"]["customJWTAuthorizer"]["discoveryUrl"]
        client_id = client_id if client_id else gateway["authorizerConfiguration"]["customJWTAuthorizer"]["allowedClients"][0]
        response = requests.get(discovery_url)
        data = response.json()
        authorization_endpoint = data["authorization_endpoint"]
        token_endpoint = authorization_endpoint.replace("/authorize", "/token")

        # Get user pool id and client secret
        user_pool_id = discovery_url.split("/")[3]
        client_secret = cognito_client.describe_user_pool_client(
            UserPoolId=user_pool_id, ClientId=client_id
        )["UserPoolClient"]["ClientSecret"]

        # Print everything
        print("User Pool ID: ", user_pool_id)
        print("Client ID: ", client_id)
        print("Client Secret: ", client_secret)

        # Connect to User Pool
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            # "scope": scope_string,
        }
        response = requests.post(token_endpoint, headers=headers, data=data)
        response.raise_for_status()
        return response.json()['access_token']

    except requests.exceptions.RequestException as err:
        print(response.json())
        return {"error": str(err)}