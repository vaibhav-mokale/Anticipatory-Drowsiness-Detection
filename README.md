# Road Hypnosis Detection System
![model related files](https://drive.google.com/drive/folders/12c9zC1uLNZ9cpo-e-xwtJVfdKDqpMpSI)
This project is a Road Hypnosis Detection System designed to alert drivers when signs of road hypnosis are detected. The system uses computer vision techniques to monitor the driver’s facial features, such as eye closure, yawning, and head tilting. If any hypnosis indicators are detected, the system triggers an alert to help prevent accidents.

![diagram](https://github.com/user-attachments/assets/cd9bed8f-9589-41fe-9d18-c42bc1c28d17)

## 📌 Features  

- **Real-time Camera Feed:** Integrated camera view within the GUI.
- **Drowsiness Detection:**
  - Eye closure detection with adjustable thresholds.
  - Yawning detection based on mouth aspect ratio.
  - Head tilt detection with customizable tilt angle threshold.
- **Visual Alerts:** Green outlines around eyes, mouth, and head tilt indicators.
- **Audible Alerts:** Plays an alert sound when drowsiness is detected.
- **Adjustable Settings:** GUI sliders to modify detection thresholds.
- **Single Window Interface:** Camera feed and controls in one window.

## Controls

- **Eye AR Threshold:** Adjust sensitivity for eye closure detection.
- **Eye Closure Frames:** Set the duration for eye closure before triggering an alert.
- **Mouth AR Threshold:** Configure sensitivity for yawning detection.
- **Head Tilt Threshold:** Define the angle range for safe head positioning.

## Theming  

This application uses a carefully chosen color palette for a professional yet visually appealing design:  

- **Palette**: `#E52020`, `#FBA518`, `#F9CB43`, and `#A89C29`  
- Sourced from [colorhunt.co](https://colorhunt.co/palette/e52020fba518f9cb43a89c29).  

## How to Run  

1. **Prerequisites**:  
   - Python 3.12.9
   - OpenCV
   - Dlib
   - NumPy
   - Tkinter
   - PIL (Pillow)
   - SciPy
   - Imutils
   - Pygame
   - and more...
   
2. **Install Dependencies**:  
   ```bash  
   bash setup.sh
   ```
   Or manually:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the Application**:  
   ```bash  
   source .venv/bin/activate
   python3 -m app
   ```  

## Build  

To create an executable file:  

1. Install `pyinstaller`:  
   ```bash  
   pip install pyinstaller  
   ```  

2. Create the `.exe` file:  
   ```bash  
   pyinstaller setup.spec --clean  
   ```  

The `.exe` file will be located in the `dist` folder.  

## Contribution  

Contributions are welcome! If you have suggestions for new features, improvements, or bug fixes, feel free to open an issue or submit a pull request.

## Credits

This project is contributed by [vaibhav-mokale](https://github.com/vaibhav-mokale)

## License  

This project is licensed under the **GPL-3.0 License**. For more details, see the [LICENSE](LICENSE) file.  
