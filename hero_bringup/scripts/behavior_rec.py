#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from std_msgs.msg import String
from hero_common.msg import predictions
import tensorflow as tf
import numpy as np

class KerasPredictionNode:
    def __init__(self, model_path, buffer_size):
        rospy.init_node('behavior_rec',anonymous=True)
        # Load the Keras model
        self.model = tf.keras.models.load_model(model_path)
        rospy.loginfo(f"Model loaded successfully from: {model_path}")
        
        self.robot_names=rospy.get_param('/robot_names',[])
        # Set up ROS topics
        self.subscribers=[]
        self.publishers={}
        self.data_buffer={}

        for robot_name in self.robot_names:
            topic_name=f"/{robot_name}/distance"
            self.subscribers.append(rospy.Subscriber(topic_name,Float32,self.input_callback,robot_name))
            self.publishers[robot_name]=rospy.Publisher(f"/prediction/{robot_name}",predictions,queue_size=10)
            self.data_buffer[robot_name] = []

       
        
        # Initialize a list to store incoming float data
        self.buffer_size = buffer_size  # Simple list for buffering
        

    def input_callback(self, msg, robot_name):
        """
        Callback for the subscribed float topic. Buffers the data and implements a sliding window.
        """
        try:
            # Add the new float value to the buffer
            self.data_buffer[robot_name].append(msg.data)
            
            # If the buffer size exceeds the maximum size, remove the oldest element
            if len(self.data_buffer[robot_name]) > self.buffer_size:
                self.data_buffer[robot_name]=self.data_buffer[robot_name][38:] # Remove the oldest value
            
            #rospy.loginfo(f"Buffered {len(self.data_buffer)}/{self.buffer_size} values.")
            
            # Check if the buffer is full
            if len(self.data_buffer[robot_name]) == self.buffer_size:
                # Run predictions on the buffered data
                self.run_prediction(robot_name)
        except Exception as e:
            rospy.logerr(f"Error buffering data: {e}")

    def run_prediction(self,robot_name):
        """
        Run predictions on the buffered data and process the result.
        """
        try:
            # Convert the buffer to a NumPy array
            input_data = np.array(self.data_buffer[robot_name]).reshape(1,self.buffer_size, 1) # Shape: (1, buffer_size)
            
            # Predict using the model
            prediction = self.model.predict(input_data)
            
            # Process the prediction result
            self.process_prediction(prediction,robot_name)
        except Exception as e:
            rospy.logerr(f"Error during prediction: {e}")

    def process_prediction(self, prediction,robot_name):
        """
        Process the raw prediction result and publish it to the result topic.
        """
        result = prediction  # Example: Argmax for classification, or adjust as needed
        #rospy.loginfo(f"Prediction result: {result},{robot_name}")
        #rospy.loginfo(prediction)
        r=predictions()
        r.x=result[0][0]
        r.y=result[0][1]
        r.z=result[0][2]
        # Publish the result
        self.publishers[robot_name].publish(r)

def main():
        
    # Node configuration
    model_path = "/home/varun/catkin_ws/src/hero_common/hero_bringup/config/behaviorpredict.keras"  # Path to your saved Keras mode
    buffer_size = 165               #30 seconds ==660         # Number of data points to collect before prediction
    
    # Instantiate the node class
    node = KerasPredictionNode(model_path,buffer_size)
    
    # Spin to keep the node running
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down KerasPredictionNode.")

if __name__ == "__main__":
    main()

