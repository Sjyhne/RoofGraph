import json
import torch
import pathlib

class RoofGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path

        # Get all building ids
        self.building_ids = self.__get_building_ids()
    
    def __get_building_ids(self):
        building_jsons = pathlib.Path(self.data_path).joinpath("buildings_transformed").glob("*.json")
        building_ids = [building_json.stem for building_json in building_jsons]
        self.num_buildings = len(building_ids)
        return building_ids

    def __load_building_json(self, building_id: str):
        building_json = pathlib.Path(self.data_path).joinpath("buildings_transformed", f"{building_id}.json")
        with open(building_json, "r") as f:
            return json.load(f)

    def __len__(self):
        return self.num_buildings
    
    def __getitem__(self, idx):
        building_id = self.building_ids[idx]
        building_json = self.__load_building_json(building_id)
        print(building_json)
        exit("")


        return self.data[idx]
    

if __name__ == "__main__":

    data_folder = "data/tromso"
    
    dataset = RoofGraphDataset(data_path=data_folder)
    dataset[0]