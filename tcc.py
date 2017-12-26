"""
Pipelines everything. From the segmentation, feature extration, preprocessing and training
"""
import segmentation as se
import batches_creation as bc

def main():
    """
    Where the magic happens
    """
    labels_path = "../dataset/labels"
    audios_path = "../dataset/audios"
    features_path = "../dataset/features"
    batches_path = "../dataset/batches"

    #Extract the features into the folder
    print("Starting features extraction.")
    se.features_from_folder(labels_path, audios_path, features_path)

    #make the batches
    bc.make_batches(features_path, batches_path)

    
if __name__ == '__main__':
    main()
