import numpy as np
from f1tenth_benchmarks.utils.BasePlanner import BasePlanner

class FollowTheGap(BasePlanner):
    def __init__(self, test_id):
        super().__init__("FollowTheGap", test_id)
        self.name = 'FollowTheGap'

    def plan(self, obs):   
        #FollowTheGap
        #1) Find nearest point 
        scan = obs['scan']
        #scanlenght = len(scan)
        # lidar scan resolution is 1080 
        proc_ranges =  self.preprocessLidar(scan)
        closest_Point = proc_ranges.argmin()

        #2) Put safety bubble around point eliminate points (set to 0) inside bubble
        min_index = closest_Point - self.planner_params.bubble_radius
        max_index = closest_Point + self.planner_params.bubble_radius
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges) - 1 
        proc_ranges[min_index:max_index] = 0  

        #3) Find max gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        #4) Find best point in gap
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        # Set steering angle
        steering_angle = self.get_angle(best, len(proc_ranges))
        if abs(steering_angle) > self.planner_params.straights_steering_angle:
            speed = self.planner_params.corners_speed
        elif abs(steering_angle) > self.planner_params.fast_steering_angle:
            speed = self.planner_params.straights_speed
        else:
            speed = self.planner_params.fast_speed

        action = np.array([steering_angle, speed])

        return action
    
    def preprocessLidar(self, ranges):
        """Preprocessing of Lidar data includes:
            1) Clipping of unnecassary points
            2)Setting each value to a mean over some window 
            3) Rejecting high values (eg. > 3m)
        """
        # Map scanned points to circular POV
        self.radians_per_elem = (2 * np.pi) / len(ranges)
        # Lidar data from behind are not required
        proc_ranges = np.array(ranges[135:-135])
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.planner_params.preprocess_conv_size), 'same') / self.planner_params.preprocess_conv_size
        proc_ranges = np.clip(proc_ranges, 0 , self.planner_params.max_lidar_dist)
        return proc_ranges
    
    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
            free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
        """
        # mask the bubble
        masked =  np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        # print(slices)
        # max_len = slices[-1].stop - slices[-1].start
        # chosen_slice = slices[-1]
        # I think we will only ever have a maximum of 2 slices but will handle an
        # indefinitely sized list for portablility

        for sl in slices[::-1]:
            #print(sl)
            sl_len = sl.stop - sl.start
            if sl_len > self.planner_params.safe_threshold:
                chosen_slice = sl
                # print("Slice choosen")
                return chosen_slice.start, chosen_slice.stop
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.planner_params.best_point_conv_size), 'same') / self.planner_params.best_point_conv_size
        return averaged_max_gap.argmax() + start_i
    
    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the lidar data and transform it into an appropriate steering angle
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2 
        steering_angle = np.clip(steering_angle, -self.planner_params.max_steer,self.planner_params.max_steer)
        return steering_angle

def main():
    #print('Hello world')
    pass

if __name__ == '__main__':
    main()