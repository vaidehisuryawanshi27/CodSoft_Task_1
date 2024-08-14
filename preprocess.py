import pandas as pd
import re

def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def clean_text(text):
    # Replace non-printable characters and control characters with a space
    cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
    return cleaned_text.strip()

def preprocess_train_data(lines):
    ids = []
    titles = []
    genres = []
    descriptions = []
    for line in lines:
        parts = line.strip().split(':::')
        if len(parts) < 4:
            print(f"Issue with line: {line}")
            continue
        
        id_ = parts[0].strip()  # Assuming ID is the first part
        title = parts[1].strip()  # Assuming TITLE is the second part
        genre = parts[2].strip()  # Assuming GENRE is the third part
        description = ":::".join(parts[3:]).strip()  # Join parts beyond index 2 as DESCRIPTION
        
        # Clean description text
        cleaned_description = clean_text(description)
        
        ids.append(id_)
        titles.append(title)
        genres.append(genre)
        descriptions.append(cleaned_description)
    
    return ids, titles, genres, descriptions

def preprocess_test_data(lines):
    ids = []
    titles = []
    descriptions = []
    for line in lines:
        parts = line.strip().split(':::')
        if len(parts) < 3:
            print(f"Issue with line: {line}")
            continue
        
        id_ = parts[0].strip()  # Assuming ID is the first part
        title = parts[1].strip()  # Assuming TITLE is the second part
        description = ":::".join(parts[2:]).strip()  # Join parts beyond index 1 as DESCRIPTION
        
        # Clean description text
        cleaned_description = clean_text(description)
        
        ids.append(id_)
        titles.append(title)
        descriptions.append(cleaned_description)
    
    return ids, titles, descriptions

def preprocess_test_solution_data(lines):
    ids = []
    titles = []
    genres = []
    descriptions = []
    for line in lines:
        parts = line.strip().split(':::')
        if len(parts) < 4:
            print(f"Issue with line: {line}")
            continue
        
        id_ = parts[0].strip()  # Assuming ID is the first part
        title = parts[1].strip()  # Assuming TITLE is the second part
        genre = parts[2].strip()  # Assuming GENRE is the third part
        description = ":::".join(parts[3:]).strip()  # Join parts beyond index 2 as DESCRIPTION
        
        # Clean description text
        cleaned_description = clean_text(description)
        
        ids.append(id_)
        titles.append(title)
        genres.append(genre)
        descriptions.append(cleaned_description)
    
    return ids, titles, genres, descriptions

def main():
    # Define your file paths here
    train_file_path = 'C:/Users/Vaidehi Suryawanshi/Downloads/movie genre classification/Genre Classification Dataset/train_data.txt'
    test_file_path = 'C:/Users/Vaidehi Suryawanshi/Downloads/movie genre classification/Genre Classification Dataset/test_data.txt'
    test_solution_file_path = 'C:/Users/Vaidehi Suryawanshi/Downloads/movie genre classification/Genre Classification Dataset/test_data_solution.txt'
    
    # Read data from files
    train_lines = read_data(train_file_path)
    test_lines = read_data(test_file_path)
    test_solution_lines = read_data(test_solution_file_path)
    
    # Preprocess data
    train_ids, train_titles, train_genres, train_descriptions = preprocess_train_data(train_lines)
    test_ids, test_titles, test_descriptions = preprocess_test_data(test_lines)
    test_solution_ids, test_solution_titles, test_solution_genres, test_solution_descriptions = preprocess_test_solution_data(test_solution_lines)
    
    # Create DataFrames
    train_df = pd.DataFrame({'ID': train_ids, 'TITLE': train_titles, 'GENRE': train_genres, 'DESCRIPTION': train_descriptions})
    test_df = pd.DataFrame({'ID': test_ids, 'TITLE': test_titles, 'DESCRIPTION': test_descriptions})
    test_solution_df = pd.DataFrame({'ID': test_solution_ids, 'TITLE': test_solution_titles, 'GENRE': test_solution_genres, 'DESCRIPTION': test_solution_descriptions})
    
    # Save DataFrames to CSV files
    save_dir = 'C:/Users/Vaidehi Suryawanshi/Downloads/movie genre classification/Genre Classification Dataset/'
    train_df.to_csv(save_dir + 'train.csv', index=False)
    test_df.to_csv(save_dir + 'test.csv', index=False)
    test_solution_df.to_csv(save_dir + 'test_data_solution.csv', index=False)

if __name__ == '__main__':
    main()
