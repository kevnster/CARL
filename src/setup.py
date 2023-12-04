import matplotlib.pyplot as plt
import random
from ai2thor.controller import Controller
from pynput import keyboard
from pynput.keyboard import Key

available_scenes = [f"FloorPlan{i}" for i in range(210, 231)]

# Randomly select a scene
random_scene = random.choice(available_scenes)

controller = Controller(
    agentMode="default",
    visibilityDistance=1.5,
    scene=random_scene,

    # step sizes
    gridSize=0.10,
    snapToGrid=False,
    rotateStepDegrees=10,

    # image modalities
    renderDepthImage=False,
    renderInstanceSegmentation=False,

    # camera properties
    width=1600,
    height=1200,
    fieldOfView=90

)


def show_image(event):
    plt.imshow(event.frame)
    plt.axis('off')
    plt.show()


def on_key_release(key):
    try:
        if key == Key.right:
            print("Rotate Right")
            event = controller.step("RotateRight")
        elif key == Key.left:
            print("Rotate Left")
            event = controller.step("RotateLeft")
        elif key == Key.up:
            print("Move Ahead")
            event = controller.step("MoveAhead")
        elif key == Key.down:
            print("Move Back")
            event = controller.step("MoveBack")
        elif key == Key.esc:
            # Stop the listener
            return False

        # Update the display after an action
        show_image(controller.last_event)

    except Exception as e:
        print(f"An error occurred: {e}")


# Set up the listener
listener = keyboard.Listener(on_release=on_key_release)

# Start the listener
listener.start()

# Keep the script running
try:
    listener.join()
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Stop the controller when done
    controller.stop()
