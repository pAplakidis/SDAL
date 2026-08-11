import carla

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

print(client.get_available_maps())
