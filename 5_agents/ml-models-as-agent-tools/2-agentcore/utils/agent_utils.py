from mcp.client.streamable_http import streamablehttp_client 


def create_streamable_http_transport(mcp_url: str, access_token: str):
       return streamablehttp_client(mcp_url, headers={"Authorization": f"Bearer {access_token}"})


def get_full_tools_list(client):
    """
    List tools w/ support for pagination
    """
    more_tools = True
    tools = []
    pagination_token = None
    while more_tools:
        tmp_tools = client.list_tools_sync(pagination_token=pagination_token)
        tools.extend(tmp_tools)
        if tmp_tools.pagination_token is None:
            more_tools = False
        else:
            more_tools = True 
            pagination_token = tmp_tools.pagination_token
    return tools