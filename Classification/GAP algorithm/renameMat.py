import os

def rename_mat(fold):
    files = [f for f in os.listdir('.') if f.endswith('.mat')]
    for file in files:
        if file == 'bestScore_evol.mat':
            new_name = f'bestScore_evol_{fold}_.mat'
            os.rename(file, new_name)

        elif file == 'meanScore_evol.mat':
            new_name = f'meanScore_evol_{fold}_.mat'
            os.rename(file, new_name)

        elif file.startswith('pop_at_generation_'):
            base_name, _ = os.path.splitext(file)
            new_name = f'{base_name}_{fold}_.mat'
            os.rename(file, new_name)

