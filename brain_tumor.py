{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c07cecd2-2027-485b-b9cc-7434e630455d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import load_model\n",
    "from PIL import Image\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fdd67eef-0952-4bfd-bb70-16fdd9435c94",
   "metadata": {},
   "outputs": [],
   "source": [
    "# load saved model\n",
    "brain_tumor_model = load_model('brain_tumor_custom_cnn.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0c00e8b9-3c62-463c-a5c5-4305576bd327",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define class names\n",
    "class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "51c45074-0641-487e-872f-ac38ab82168d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-07-20 13:22:13.084 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-20 13:22:13.773 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\uttam\\brain-tumor-env\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-07-20 13:22:13.775 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-20 13:22:13.776 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-20 13:22:13.777 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-20 13:22:13.778 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-20 13:22:13.780 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# Streamlit UI\n",
    "st.title('Brain Tumor MRI Classification')\n",
    "st.write('Upload an MRI image to predict tumor type.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "33dd21bd-4395-4274-ba54-458c285c411c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-07-20 13:23:26.406 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-20 13:23:26.414 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-20 13:23:26.416 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-20 13:23:26.417 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-20 13:23:26.418 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-20 13:23:26.419 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# Image upload \n",
    "upload_file = st.file_uploader(\"Choose and MRI Image...\",type = ['jpg','jpeg','png'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "48f3b4b1-532c-4833-9bd9-a91cded33718",
   "metadata": {},
   "outputs": [],
   "source": [
    "if upload_file is not None:\n",
    "    image = Image.open(upload_file).convert('RGB')\n",
    "    st.image(image, caption = 'Uploaded Image', user_column_width = True)\n",
    "\n",
    "    # Preprocess Image\n",
    "    img = image.resize((224,224))\n",
    "    img_array = np.array(img) / 255.0\n",
    "    img_array = np.expand_dims(img_array, axis=0)\n",
    "\n",
    "    # Predict\n",
    "    prediction = brain_tumor_model.predict(img_array)\n",
    "    predicted_class = class_names[np.argmax(prediction)]\n",
    "\n",
    "    #Output\n",
    "    st.markdown(f'### Predicted Tumor Type: ,**{predicted_class.upper()}**')\n",
    "    st.bar_chart(prediction[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a19548c6-d2e7-4dae-a6d7-2cadeeff8079",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (3433106372.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[8], line 1\u001b[1;36m\u001b[0m\n\u001b[1;33m    streamlit run brain_tumor.ipy\u001b[0m\n\u001b[1;37m              ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "e25d523f-e745-41a5-b76e-5cfa2afa7ed5",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
