import json
import os
import numpy as np

data_type = 'val'
data_path = '../{}_data/data'.format(data_type)

# combine event data
whole_data = []

for participant in os.listdir(data_path):
    person_path = '{}/{}'.format(data_path, participant)
    event_file_path = '{}/{}_events.json'.format(person_path, participant)
    with open(event_file_path) as json_file:
        data = json.load(json_file)
    for event in data:
        if event['event'] == 'new_stage':
            event['event'] = 0
        elif event['event'] == 'flag_reached':
            event['event'] = 1
        elif event['event'] == 'status_up':
            event['event'] = 2
        elif event['event'] == 'status_down':
            event['event'] = 3
        else:
            event['event'] = 4
        
        event['player'] = participant

    whole_data = whole_data + data
    print('-----------')
    print(whole_data)

output_file_path = '../{}_data/event/event.json'.format(data_type)
file = open(output_file_path, "w")
json.dump(whole_data, file)
file.close()


# frame event label
# for participant in os.listdir(data_path):
#     person_path = '{}/{}'.format(data_path, participant)
#     game_path = '{}/{}_game'.format(person_path, participant)

#     frame_label = np.zeros(len(os.listdir(game_path))+1)
    
#     event_file_path = '{}/{}_events.json'.format(person_path, participant)
#     with open(event_file_path) as json_file:
#         data = json.load(json_file)

#     for event in data:
#         frame_label[event['frame_number']] = 1

#     output_file_path = '../test_data/data/{}/{}_event_ornot.npy'.format(participant, participant)
#     np.save(output_file_path, frame_label)
    
