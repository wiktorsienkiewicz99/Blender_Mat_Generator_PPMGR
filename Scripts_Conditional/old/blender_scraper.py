import pyautogui
import keyboard
import time

# List to store click positions
click_positions = []
# Flag to control the clicking loop
continue_clicking = True
# Flag to control pausing
is_paused = False

print("Hover over a point and press 'p' to save the position. Press 'q' to start clicking on saved positions. Press 'Enter' to pause or resume.")

def save_click_position():
    x, y = pyautogui.position()
    click_positions.append((x, y))
    print(f"Position saved: {x}, {y}")

def toggle_pause():
    global is_paused
    is_paused = not is_paused
    if is_paused:
        print("Paused.")
    else:
        print("Resumed.")

# Listen for 'p' to save positions
keyboard.add_hotkey('p', save_click_position)
# Listen for 'Enter' to toggle pausing
keyboard.add_hotkey('enter', toggle_pause)

# Wait for the user to press 'q' to start the clicking loop
keyboard.wait('q')

print(f"Starting loop of clicks on {len(click_positions)} saved positions. Press 'Enter' to pause or resume.")

while continue_clicking:
    for position in click_positions:
        # Check if paused
        while is_paused:
            time.sleep(10)  # Pause execution without exiting the loop
        pyautogui.moveTo(position[0], position[1], duration=0.1)
        pyautogui.click()
        print(f"Clicked at {position}")
        time.sleep(3)  # Adjust this delay as needed

print("Stopped clicking.")
q