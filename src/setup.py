from ai2thor.controller import Controller
import matplotlib.pyplot as plt

# Initialize the Controller
controller = Controller(
    agentMode="default",
    scene="FloorPlan23",
    gridSize=0.25,
    snapToGrid=True,
    rotateStepDegrees=90,
    width=1280,
    height=720,
    fieldOfView=90
)

def show_image(event):
    plt.imshow(event.frame)
    plt.axis('off')
    plt.show()

try:
    while True:
        # Display the current view
        show_image(controller.last_event)

        # Get input from the user
        action = input("Enter action (MoveAhead, RotateRight, RotateLeft, MoveBack, Exit): ")

        # Perform the action
        if action in ["MoveAhead", "RotateRight", "RotateLeft", "MoveBack"]:
            event = controller.step(action)
        elif action == "Exit":
            break
        else:
            print("Invalid action.")

except KeyboardInterrupt:
    pass

# Stop the controller when done
controller.stop()
