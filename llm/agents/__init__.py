
def load_react_shots(ds_name='react_datasets'):
    import yaml

    # Read YAML 
    with open('react_shots.yaml', 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    # load data
    react_datasets = data[ds_name]

    # print text
    for dataset in react_datasets:
        print(f"{dataset['text']}\n")
