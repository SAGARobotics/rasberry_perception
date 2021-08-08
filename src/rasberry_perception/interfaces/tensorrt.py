from threading import Event
import ros_numpy
import json
from rasberry_perception.interfaces.default import BaseDetectionServer
from rasberry_perception.interfaces.registry import DETECTION_REGISTRY
from rasberry_perception.msg import Detections, ServiceStatus, RegionOfInterest, SegmentOfInterest, Detection
from rasberry_perception.srv import GetDetectorResultsResponse, GetDetectorResultsRequest
from rasberry_perception.utility import function_timer

@DETECTION_REGISTRY.register_detection_backend("TensorrtServer")
class TensorrtServer(BaseDetectionServer):
    def __init__(self, config_path, service_name, image_height=480, image_width=640, image_hz=30):    
        try:
            import modularmot
            from modularmot.utils import ConfigDecoder
            import os
        except ImportError:
            raise
        with open(config_path) as cfg_file:
            config = json.load(cfg_file, cls=ConfigDecoder)        
        project_dir = os.path.dirname(__file__)
        self.nms_max_overlap = nms_max_overlap        
        
        print("Load Engine")
        self.mot = modularmot.MOT([int(image_width), int(image_height)],1.0/int(image_hz), config['mot'], detections_only=True, verbose=False)
        self.currently_busy = Event()
        
        # Base class must be called at the end due to self.service_server.spin()
        BaseDetectionServer.__init__(self, service_name=service_name)

    @staticmethod
    def citation_notice():
        return "TensorRT Inference\n" \
               "Maintained by Robert Belshaw (rbelshaw@sagarobotics.com)"

    @function_timer.interval_logger(interval=10)
    def get_detector_results(self, request):
        """
        Args:
            request (GetDetectorResultsRequest):
        Returns:
            GetDetectorResultsResponse
        """
        try:
            import numpy as np
        except ImportError:
            raise
        
        if self.currently_busy.is_set():
            return GetDetectorResultsResponse(status=ServiceStatus(BUSY=True))
        self.currently_busy.set()
        detections_msg = Detections()        
        try:
            image = ros_numpy.numpify(request.image)
            if request.image.encoding == "rgb8":
                image = image[..., ::-1]

            self.mot.step(image)
            for detection in self.mot.detections:
                x1, y1, x2, y2 = detection[0][0], \
                detection[0][1], \
                detection[0][2], \
                detection[0][3]
                if detection[1] == 0:
                    class_ = "Ripe Strawberry"
                else:
                    class_ = "unripe"
                detections_msg.objects.append(Detection(roi=RegionOfInterest(x1=x1, y1=y1, x2=x2, y2=y2), seg_roi=SegmentOfInterest(x=[], y=[]), id=self._new_id(), track_id=-1, confidence=detection[2], class_name=class_))                        
        except Exception as e:
            self.currently_busy.clear()
            print("TensorrtServer error: ", e)
            return GetDetectorResultsResponse(status=ServiceStatus(ERROR=True), results=detections_msg)
        self.currently_busy.clear()
        return GetDetectorResultsResponse(status=ServiceStatus(OKAY=True), results=detections_msg)

