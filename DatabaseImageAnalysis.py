import numpy as np
import scipy.spatial
import torch
import torchvision
import matplotlib.pyplot as plt
from IPython.display import display, HTML

def task1a():
    data = np.genfromtxt('csvdata/Wholesale customers data.csv', 
                         delimiter=',', 
                         names=True, 
                         usecols=np.arange(2, 8))
    
    features = data.dtype.names
    for feature in features:
        mean_value = data[feature].mean()
        median_value = np.median(data[feature])
        print(f"{feature:20s} {mean_value:10.3f} {median_value:10.3f}")

def task1b():
    data = np.genfromtxt('csvdata/CortexNuclear.csv', 
                         delimiter=',', 
                         skip_header=1, 
                         usecols=np.arange(1, 78))
    clean_data = data[~np.isnan(data).any(axis=1)]
    plt.imshow(clean_data[:30])

def task2a(db):
    cursor = db.cursor()
    query = '''SELECT genres.Name, tracks.Milliseconds
               FROM tracks 
               INNER JOIN genres ON genres.GenreID = tracks.GenreID;'''
    
    result = np.array(cursor.execute(query).fetchall())
    genres = result[:, 0]
    duration = result[:, 1].astype(float) / 1000.0
    
    for genre in sorted(set(genres)):
        genre_duration = duration[genres == genre].mean()
        print(f"{genre:20s} {genre_duration:8.3f}")

def task2b(db):
    cursor = db.cursor()
    query = '''SELECT genres.Name, customers.Country
               FROM invoice_items
               INNER JOIN invoices ON invoice_items.InvoiceId = invoices.InvoiceId
               INNER JOIN customers ON customers.CustomerId = invoices.CustomerId
               INNER JOIN tracks ON invoice_items.TrackId = tracks.TrackId
               INNER JOIN genres ON genres.GenreId = tracks.GenreId;'''
    
    result = np.array(cursor.execute(query).fetchall())
    genres = result[:, 0]
    countries = result[:, 1]
    
    unique_genres = sorted(set(genres))
    unique_countries = sorted(set(countries))
    
    print(f"{'':15s}" + "".join([f"{c[:3]:3s}" for c in unique_countries]))
    
    for genre in unique_genres:
        genre_counts = []
        for country in unique_countries:
            count = len(result[(genres == genre) & (countries == country)])
            genre_counts.append(f"{count:3d}")
        print(f"{genre[:15]:15s}" + "".join(genre_counts))

def task3a(imagesresize):
    images_array = np.array(imagesresize)
    distance_matrix = scipy.spatial.distance.cdist(images_array, images_array)
    return distance_matrix

def task3b(images, model, normalize):
    feature_vectors = []
    
    for img in images:
        img_tensor = torchvision.transforms.ToTensor()(img).unsqueeze(0)
        features = model.forward(normalize(img_tensor))
        feature_vectors.append(features.data.numpy()[0].mean(axis=2).mean(axis=1))
    
    feature_vectors_array = np.array(feature_vectors)
    distance_matrix = scipy.spatial.distance.cdist(feature_vectors_array, feature_vectors_array)
    return distance_matrix
