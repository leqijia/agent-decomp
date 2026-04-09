import json
class output_mapper:
    field_map = {
        "City": "city_name",
        "cty": "city_name",
        "city": "city_name",
        "metropolis" : "city_name",
        "Cty":"city_name"
    }
    def normalize(self,data):
        values = [None]*(len(data))
        i = 0
        for record in data:
            for key, value in record.items():
                output_field_name = self.field_map.get(key)
                if output_field_name is not None:
                    print(output_field_name + " : " + value)
            