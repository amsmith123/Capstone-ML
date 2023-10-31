import pickle

# Script to read a .pyb file and write its content to standard output
def read_and_output_pyb(in_file_path,out_file_path):
    try:
        with open(in_file_path, 'rb') as file:
    # Load (deserialize) the content
            content = pickle.load(file)
         # Open an output file in binary mode
            with open(out_file_path, 'w') as outfile:
                # Write the content to the output file
                j = []
                i = 0
                for items in content:
                    for item in items:
                        outfile.write(str(item) + '\n')
                        i += 1
                    j.append(i)
                    i = 0
                print(f"Number of items: {str(j)}")
            
    except FileNotFoundError:
        print(f"Error: File {in_file_path} not found.")
    except PermissionError:
        print(f"Error: Permission denied for {in_file_path}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

i = 0
file_path = f'Data/split_weakly_{i}.pyb'
out_file = f'split_weakly_{i}.txt'
read_and_output_pyb(file_path, out_file)
