import json
""" from  webarena.browser_env.actions import create_id_based_action
from  webarena.browser_env.actions import ActionParsingError """
class output_mapper:
    def __init__(self, success_file, error_file):
        self.success_file = success_file
        self.error_file = error_file

        self.stop_reason_values = { "agent_stop","max_steps","parse_failures","repeating_action","env_terminated","crash"}
        self.field_map = {
                    "task_id": "task_id",
                    "config_file": "config_file",
                    "sites": "sites",
                    "intent": "intent",
                    "start_url": "start_url",
                    "model": "modle",
                    "agent_variant": "agent_variant",
                    "observation_type": "observation_type",
                    "action_set_tag": "action_set_tag",
                    "max_steps": "max_steps",
                    "started_at": "started_at",
                    "ended_at":   "ended_at",
                    "total_steps": "total_steps",
                    "stop_reason": "stop_reason",
                    "success": "success",
                    "eval_score": "eval_score",
                    "t": "t",
                    "url": "url",
                    "observation": "observation",
                    "thought": "thought",
                    "raw_prediction": "raw_prediction",
                    "action": "action",
                    "parse_error": "parse_error",
                    "steps":"steps"
        }
        
    def normalize(self,data):
        success_data = {}
        error_data = {}
        for key, value in data.items():
            if key in self.field_map:
                #print(key , ":" , value)
                if isinstance(value, list) :
                   s,e = self.handle_steps_array(key,value) 
                   if s:
                      success_data[key] = s
                   if e:
                       error_data[key] = e 
                else:
                    b = self.validate_stop_reason(key,value)
                    b &= self.validate_total_steps(data,key)
                    if b:
                        success_data[key] = value
                    else:
                        error_data[key] = value
        self.write_json(self.success_file, success_data)
        self.write_json(self.error_file, error_data,)
        self.summarize(success_data,error_data)        
                

    def handle_steps_array(self, key, value):
        success_steps = []
        error_steps = []
        if key.lower() != "steps":
            return  success_steps,error_steps
        index = 1
        for record in value:
            success_json = {}
            error_json = {}
            for record_key , record_value in record.items():
                output_field_name = self.field_map.get(record_key)
                if output_field_name is not None:
                    p = self.validate_steps_t(record_key, record_value, index)
                    p &= self.validate_stop_reason(record_key,record_value)
                    p &= self.validate_action(record,record_key)
                    if p:
                        success_json[record_key] = record_value
                    else:
                        error_json[record_key] = record_value
            index += 1
            if success_json:
                success_steps.append(success_json)
            if error_json:
                error_steps.append(error_json)

        return success_steps, error_steps

    
    def validate_stop_reason(self,key,value):
            if key != "stop_reason":
                return True
            else:
                return value in self.stop_reason_values

    def validate_action(self,data,key):
       if key != "action":
           return True
       action = data.get("action","")
       parse_error = data.get("parse_error",None)
       if (action == "") == (parse_error is not None):
           return True
       else:
           return False
    
    def validate_total_steps(self,data,key):
        if key != "total_steps":
            return True
        else:
            total_steps = data.get("total_steps","")
            array_length= len((data.get("steps"),None)) 
            if int(total_steps) == array_length:
                return True
            else: 
                return False

    def validate_steps_t(self, record_key, record_value, index):
        if record_key != 't':
            return True
        return (record_value == index)
    
    def write_json(self, file_name, data):
        if not data:
            print(f"Skipping file {file_name} (no data)")
            return

        with open(file_name, "w", encoding="utf-8") as f:
         json.dump(data, f, indent=2)
   
    def summarize(self, success_data , error_data):
        b_success = bool(success_data)
        b_error = bool(error_data)
        if b_success == False and b_error == False:
            print("No avalible data")
            return
        elif b_success == True and b_error == False:
            print("All success"+ self.success_file)
        elif b_success == False and b_error == True:
            print("There are errors in the file"+self.error_file)
        elif b_success == True and b_error == True:
            print("Partial success. Success file: "+ self.success_file + " Error file: " + self.error_file) 