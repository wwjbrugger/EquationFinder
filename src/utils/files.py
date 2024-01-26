def create_file_path(save_folder, stem):
    id = uniqu_id(save_folder, stem)
    file_path = save_folder / f'{stem}_{id}'
    return file_path


def uniqu_id(path, stem):
    p = path.glob('*')
    folders = [x for x in p if x.is_dir()]
    max_number = 0
    for folder in folders:
        try:
            run_number = int(str(folder.stem).replace(f"{stem}_",''))
            if run_number > max_number:
                max_number = run_number
        except ValueError:
            pass


    return max_number + 1

def highest_number_in_files(path,stem):
    p = path.glob('*')
    files = [x for x in p if x.is_file()]
    max_number = 0
    for file in files:
        try:
            run_number = int(str(file.stem).replace(f"{stem}", ''))
            if run_number > max_number:
                max_number = run_number
        except ValueError:
            pass

    return max_number