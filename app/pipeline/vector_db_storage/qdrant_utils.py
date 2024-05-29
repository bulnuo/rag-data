
def collection_exists(client, collection_name):
    try:
        client.get_collection(collection_name=collection_name)
        return True
    except Exception as e:
        return False

