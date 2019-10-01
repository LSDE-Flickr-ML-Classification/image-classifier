import pycuda.driver as cuda
import pycuda.autoinit

cuda.init()

class AboutCudaDevices():

    @staticmethod
    def num_devices():
        """Return number of devices connected"""
        return cuda.Device.count()

    @staticmethod
    def devices():
        """Get info on all devices connected"""
        num = cuda.Devce.count
        print("%d device(s) found" % num)
        for i in range(num):
            print(cuda.Device(i).name(), "(Id: %d)" % i)

    @staticmethod
    def mem_info():
        """Get available and total memory of all devices"""
        available, total = cuda.mem_get_inf()
        print("Available: %.2f GB \n Total: %.2f GB" % (available / 1e9, total / 1e9))

    @staticmethod
    def attributes(device_id=0):
        """Get attributes of device with devie Id = device_id"""
        return cuda.Device(device_id).get_attributes()

    def __repr__(self):
        """Class representation as number of devices connected and about them."""
        num = cuda.Device.count()
        string = ""
        string += ("%d device(s) found:\n" % num)
        for i in range(num):
            string += ("    %d) %s (Id: %d)\n" % ((i + 1), cuda.Device(i).name(), i))
            string += ("          Memory: %.2f GB\n" % (cuda.Device(i).total_memory() / 1e9))
        return string


cuda_info = AboutCudaDevices()
print(cuda_info)
