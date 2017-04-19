import cv2


class MotionDetector(object):
    def __init__(self, min_area):
        self.prev_frame = None
        self.min_area = min_area

    def get_mass_point(self, motions):
        """
        Calculate a weighted average of all points.
        """
        if not motions:
            return None
        
        total_wight = sum(m[4] for m in motions)

        x = sum(
            (m[0] + m[2] / 2) * m[4]
            for m in motions
        ) / total_wight

        y = sum(
            (m[1] + m[3] / 2) * m[4]
            for m in motions
        ) / total_wight

        return x, y

    def submit_frame(self, frame):
        if len(frame.shape) == 3 and frame.shape[2]  > 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = cv2.GaussianBlur(frame, (21, 21), 0)

        prev_frame = self.prev_frame
        self.prev_frame = frame

        if prev_frame is None:
            return {}


        frame_delta = cv2.absdiff(prev_frame, frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        res = cv2.findContours(
            thresh.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Response is version-specific
        if len(res) == 2:
            cnts, _ = res
        else:
            _, cnts, _ = res

        motions = []

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            motions.append((x, y, w, h, area))

        return {
            'motions': motions,
            'center': self.get_mass_point(motions),
        }

