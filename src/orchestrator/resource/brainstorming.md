Phani: 

- Navigation Targets TODO: 
    - Fridge Opening Target
    - Fridge Closing Target
    - Fridge Placing Target
    - Table Grasping Target
---

Dionisis:

- Motion Recorder Skills:
    - Open Fridge
    - Close Fridge
    - Pick Objects (to be updated later after we decide objects)
    - Place Objects (to be updated later after we decide objects)

For this we will do the following approach:

You publish to me under
`motion_recorder/target_skill`

A string with a STRICT PREDETERMINED name of the skill


e.g.:
"open_fridge"
"grasp_milk"

After I'm done I will return a 
/motion_recorder/status

if the string is done, then it's a success otherwise I will return the error for logging purposes, and perhaps retrying and moving away from a linear FSM (TBD if we need to do services in the end for the parliament, I think it's the best appraoch.)



> IMPORTANT!!!! 
>
> For each of my skills I need a specific frame or pose to be published before, for example
> for the fridge opening and closing I need "apriltag_2" frame to exist
> For grasping any object I need the pose of the object to be published under `/object_pose `
> And for placing the object I again need the mustard to be detected under `/object_pose`
> That being said, before asking me to close the fridge, you need to ask raphael's foujndation pose node to detect the mustard!!!!
> The apriltag node is running contiously, so no worries about asking someone for something, you can just have a dummy "node" in your fsm that checks if this is detected, we could potentially rotate the head until it is.



Reminder:
All my skills are expressed relative to a frame, that is either the object I'll grasp or an apriltag. If that is not detected, I cannot rollout, so always remember that a state before my skill state should be the detection of that object. (Or the existense of the apriltag)


--- 
Raphael:

- 6D Pose targets (Raphael, Foundation Pose) TODO: Speak with Raphael:
    - Mustard
    - Box
    - Red bottle
    - ... 




---

Overall plan of the demo

1. Robot starts in the middle of the room
2. Robot navigates to the fridge opening navigation goal
3. Check if apriltag_2 frame is detected
4. Trigger motion_recorder fridge open skill
5. Go to table & Homing position with a small start delay
6. Get pose for object i 
7. Trigger grasp object i 
8. Go to fridge placing position & Homing position with a small start delay
9. Get pose for mustard (Anchor point in fridge that defines dionisis' placing motions)
10. Trigger place object i 
11. Go to step 5 and repeat until all objects are done (Predefined sequence of objects)
12. Go to intermidiate fridge close position & Homing position with a small start delay
13. Go to Fridge close position
14. Check if apriltag_2 frame is detected
15. Trigger Close fridge motion
16. End :D


Room for improvement, if a pose detection fails, maybe we could move the head... but I don't think it is necessary for this hackathon, keep it for brainstorming - closing discussion.


apriltag_2 - Fridge April Tag 
Topic apriltag_pose/pose_tag_2 PoseStampted