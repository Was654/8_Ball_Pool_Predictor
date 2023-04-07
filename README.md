## 8 Ball Pool Predictor Contest 
### Organized by CVZone Sponsored by NVIDIA
##### Contest Task: Predict the path of the Pool Ball       
Task that needs to  be completed to get the Prediction Model
1) Detecting The Coloured Balls, Cue and the Cue Ball:
- To solve this I have used HSV Colour Spacing to extract the mask of the desired colour. Then contours were extracted from the mask to detect the Ball, Colour Balls and Cue. To differentiate between Cue and Cue Ball of being same colours various aspects like radius, width and area of the contour was considered.       
-Edge Case: Edge case was to detect the Green Balls Behind the green background.Tweaking the value of Green HSV Mask comparing the Shades of the Green Ball and Background a contour of small radius within the pool table range was considered which was sufficient enough to detect the Balls
2) Detecting the Path of  Cue Ball and Colored Ball Needed for detection of  the path for Coloured Ball:
- The connecting straight line of the Cue and Cue ball is the Path of Cue Ball

3) Detecting the Path of the Coloured Ball:
-For this a Virtual ball of greater radius than the Colored Ball was considered. And the Collision between the Cue Ball and the Virtual Coloured Ball was noted. The Collision point and the centre of the Coloured Ball is the Path of the Coloured Ball.    
#### Scenarios of the path of the Colour Ball.
- Falling into Pocket: If the Ball falls into the pocket the predictions shows In
- Hiting The Wall: If the Ball Hits the Wall then the reflected path of the Colour Ball was taken into account and jugdement was taken if the Coloured Ball goes into the   Pocket or Hits the Wall again. Depending on the outcome the prediction is shown.     
- Egde Cases: As some balls consisted of Non Solid Pool Balls and Green balls. So Radius of the balls were different and tunning was done according to respective colour.
4) Making the Prediction Path Static:
- For making the prediction Path Static all the prediction were done with the First Frame of the Individual Coloured Ball Appearance in the Pool Table. Through out the whole shot the prediction of the first appearance of the Coloured Ball was displayed. When the next Shot in the footage apprears, the prediction stored values were reseted and the prediction of the new shot was shown.
