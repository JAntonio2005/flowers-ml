# flowers_train.py  (entrena y deja el modelo listo para tu backend)
import os, json, tensorflow as tf
from tensorflow import keras as K

print("TensorFlow:", tf.__version__)

# 1) Dataset de flores (descarga automática oficial de TF)
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = K.utils.get_file('flower_photos', origin=dataset_url, untar=True)
DATA_DIR = os.path.join(os.path.dirname(data_dir), "flower_photos")

# 2) Config
IMG = 224
BATCH = 32
EPOCHS = 5  # súbelo luego si quieres más accuracy

# 3) Cargar train/val (80/20)
train = K.preprocessing.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=1337,
    image_size=(IMG, IMG), batch_size=BATCH, label_mode="int"
)
val = K.preprocessing.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=1337,
    image_size=(IMG, IMG), batch_size=BATCH, label_mode="int"
)

class_names = train.class_names  # orden de índices
print("CLASES:", class_names)

# 4) Prepro + augment
AUTOTUNE = tf.data.AUTOTUNE
aug = K.Sequential([
    K.layers.RandomFlip("horizontal"),
    K.layers.RandomRotation(0.05),
    K.layers.RandomZoom(0.1),
])

def prep(ds, training):
    ds = ds.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y))
    if training: ds = ds.map(lambda x,y: (aug(x, training=True), y))
    return ds.prefetch(AUTOTUNE)

train = prep(train, True)
val   = prep(val, False)

# 5) Modelo (MobileNetV2 congelada + cabeza)
base = K.applications.MobileNetV2(input_shape=(IMG,IMG,3), include_top=False, weights="imagenet")
base.trainable = False
inp = K.Input((IMG,IMG,3))
x = base(inp, training=False)
x = K.layers.GlobalAveragePooling2D()(x)
x = K.layers.Dropout(0.2)(x)
out = K.layers.Dense(len(class_names))(x)  # logits
model = K.Model(inp, out)
model.compile(optimizer="adam",
              loss=K.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# 6) Entrenar
model.fit(train, validation_data=val, epochs=EPOCHS)

# 7) Guardar etiquetas y modelo en rutas que usa TU backend
os.makedirs("data", exist_ok=True)
with open("data/index.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False)

os.makedirs("models", exist_ok=True)
model.save("models/saved_model")   # <- SavedModel que tu FastAPI carga

print("✅ Listo: models/saved_model  /  data/index.json")
print("⚠️ Recuerda arrancar el backend con INPUT_SIZE=224")
