import argparse
from data_loader import get_data_generators
from models import build_waste_classifier
import tensorflow as tf

def train_model(epochs, batch_size, model_save_path):
    train_gen, val_gen = get_data_generators(batch_size=batch_size)
    model = build_waste_classifier(num_classes=train_gen.num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen
    )
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a waste classification model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--model_save_path', type=str, default='trained_models/waste_sorter_model.h5', help='Path to save the trained model.')
    args = parser.parse_args()

    train_model(args.epochs, args.batch_size, args.model_save_path)