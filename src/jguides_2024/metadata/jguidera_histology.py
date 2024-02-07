import datajoint as dj
import numpy as np


schema = dj.schema('jguidera_histology')


@schema
class ValidShank(dj.Manual):
    definition = """
    # Estimated distances from probe tip to top and bottom of targeted region
    subject_id : varchar(40)
    electrode_group_name : varchar(40)
    ---
    below_dorsal_limit_lens : blob
    below_ventral_limit_lens : blob
    """

    def insert_defaults(self):
        for subject_id, v in self.get_shank_map().items():
            for electrode_group_name, lens_map in v.items():
                self.insert1(
                    {**{"subject_id": subject_id, "electrode_group_name": electrode_group_name}, **lens_map},
                    skip_duplicates=True)

    @staticmethod
    def get_shank_map():

        # Order of shanks: left to right, where left is the viewer's left when facing the probe contacts.
        # Numbers directly typed in array correspond to shanks proceeding anterior to posterior in the brain.
        # These are reversed as needed so that numbers in the dictionary proceed from left to right as defined
        # above.

        return {
            "J16": {
                # left hemisphere OFC probe:
                # maintain typed order to get left to right shank order since implanted in
                # left hemisphere with opposite side stereotax arm
                #
                "24":
                    {"below_dorsal_limit_lens": np.asarray([1, 1, .9, .92]) * 1000,
                     "below_ventral_limit_lens": np.zeros(4) * 1000},
                # right hemisphere mPFC probe:
                # reverse typed order to get left to right shank order since implanted in
                # right hemisphere with opposite side stereotax arm
                #
                "25":
                    {"below_dorsal_limit_lens": np.ones(4)[::-1] * 1000,
                     "below_ventral_limit_lens": np.zeros(4)[::-1] * 1000}},
            "mango": {
                # right hemisphere OFC probe:
                # maintain typed order to get left to right shank order since implanted in
                # right hemisphere with same side stereotax arm
                "25":
                    {"below_dorsal_limit_lens": np.asarray([1, .79, .4, .48]) * 1000,
                     "below_ventral_limit_lens": np.zeros(4)},
                # left hemisphere mPFC probe:
                # reverse typed order to get left to right shank order since implanted in
                # left hemisphere with same side stereotax arm
                "24":
                    {"below_dorsal_limit_lens": np.asarray([.77, .8, .71, .8][::-1]) * 1000,
                     "below_ventral_limit_lens": np.zeros(4)[::-1] * 1000}},
            "june": {
                # right hemisphere OFC probe:
                # reverse typed order to get left to right shank order since implanted in
                # right hemisphere with opposite side stereotax arm
                "24":
                    {"below_dorsal_limit_lens": np.ones(4)[::-1] * 1000,
                     "below_ventral_limit_lens": np.asarray([0, 0, 1, 1])[::-1] * 1000},
                # right hemisphere mPFC probe:
                # reverse typed order to get left to right shank order since implanted in
                # right hemisphere with opposite side stereotax arm
                "25":
                    {"below_dorsal_limit_lens": np.ones(4)[::-1] * 1000,
                     "below_ventral_limit_lens": np.zeros(4)[::-1] * 1000},
                # left hemisphere OFC probe:
                # maintain typed order to get left to right shank order since implanted in
                # left hemisphere with opposite side stereotax arm
                "26":
                    {"below_dorsal_limit_lens": np.ones(4) * 1000,
                     "below_ventral_limit_lens": np.asarray([0, 0, 1, 1]) * 1000}},
            "fig": {
                # left hemisphere OFC probe:
                # note no recordings on most posterior shank due to broken electrical traces
                # reverse typed order to get left to right shank order since implanted in
                # left hemisphere with same side stereotax arm
                "24":
                    {"below_dorsal_limit_lens": np.asarray([1, .95, 1, 0][::-1]) * 1000,
                     "below_ventral_limit_lens": np.zeros(4)[::-1] * 1000}},
                # dead right hemisphere OFC probe: nothing listed since dead
            "peanut": {
                # left hemisphere OFC probe:
                # maintain typed order to get left to right shank order since implanted in
                # left hemisphere with opposite side stereotax arm
                "25":
                    {"below_dorsal_limit_lens": np.asarray([1, 1, 1, .6]) * 1000,
                     "below_ventral_limit_lens": np.zeros(4) * 1000},
            }}


class LivermoreD2:

    # Specifications for Lawrence Livermore National Labs (LLNL) D2 polymer probe device
    # Measurements from NIHBrain-v2[1]_polymer_probe_specs.pdf

    def __init__(self):
        self.num_shanks = 4
        self.electrode_pitch = 26  # edge-to-edge distance in microns
        self.num_contacts_per_shank = 32
        self.electrode_diameter = 15  # microns
        # distance from deepest (most ventral) edge of deepest contact to end of probe taper, in microns
        self.electrode_edge_to_tip_dist = 135
        self.tip_to_middle_of_top_contact_dist = self._get_tip_to_middle_of_top_contact_dist()
        self.tip_to_middle_of_lowest_contact_dist = self._get_tip_to_middle_of_lowest_contact_dist()

    def _get_tip_to_middle_of_top_contact_dist(self):
        # D2 probe extent (from very tip of taper to middle of highest (most dorsal) contact)
        return self.electrode_edge_to_tip_dist + self.electrode_pitch*self.num_contacts_per_shank + \
               self.electrode_diameter / 2

    def _get_tip_to_middle_of_lowest_contact_dist(self):
        return self.electrode_edge_to_tip_dist + self.electrode_diameter / 2

    def get_num_contacts_below_dorsal_limit(self, below_dorsal_limit_len):
        """
        Get number of contacts that are below the dorsal limit of valid zone
        :param below_dorsal_limit_len: distance (in mm) of probe from shank tip upwards that is below dorsal limit
         of valid zone
        :return: number of contacts below dorsal limit of valid zone
        """

        # Find how many contacts fit inside space between deepest electrode on shank and dorsal limit of valid zone
        num_valid_contacts = int(np.floor(
            (below_dorsal_limit_len - self.electrode_edge_to_tip_dist) /  # subtract off contactless shank tip
            self.electrode_pitch))  # divide by distance between electrodes
        # return what is smaller: number of contacts on shank, or the above
        return np.min([num_valid_contacts, self.num_contacts_per_shank])

    def get_num_invalid_dorsal_contacts(self, below_dorsal_limit_len):
        return self.num_contacts_per_shank - self.get_num_contacts_below_dorsal_limit(below_dorsal_limit_len)

    def get_num_invalid_ventral_contacts(self, below_ventral_limit_len):
        """
        Get number of ventral-most contacts that are out of valid zone
        :param below_ventral_limit_len: distance (in mm) of probe from shank tip upwards that is out of valid zone
        :return: number of ventral-most contacts out of valid zone
        """

        # Invalid ventral contacts are defined as those whose centers are less than or equal to
        # below_ventral_limit_len from the tip of the shank
        num_contacts = int(np.floor(
            (below_ventral_limit_len - self.electrode_edge_to_tip_dist) /  # subtract off contactless shank tip
            self.electrode_pitch))  # divide by distance between electrodes
        return np.max([0, num_contacts])

    def get_valid_idxs(self, below_dorsal_limit_lens, below_ventral_limit_lens,
                       electrode_order="dorsal_to_ventral"):
        # Get valid idxs in list of electrodes based on a valid lower range length

        # Check inputs
        # check electrode order type valid
        if electrode_order != "dorsal_to_ventral":
            raise Exception(f"Code currently only supports dorsal_to_ventral electrode_order")
        # check ranges valid
        for k, v in {"below_dorsal_limit_lens": below_dorsal_limit_lens,
                     "below_ventral_limit_lens": below_ventral_limit_lens}.items():
            if len(v) != self.num_shanks:
                raise Exception(f"{k} must have exactly as many elements as self.num_shanks: {self.num_shanks}")

        # Get number of invalid dorsal channels for each shank
        num_invalid_dorsal_contacts = [self.get_num_invalid_dorsal_contacts(x) for x in below_dorsal_limit_lens]

        # Get number of invalid ventral channels for each shank
        num_invalid_ventral_contacts = [self.get_num_invalid_ventral_contacts(x) for x in below_ventral_limit_lens]

        # Return valid idxs
        # ...If list of electrodes proceeds dorsal to ventral
        if electrode_order == "dorsal_to_ventral":
            return np.concatenate([
                np.arange(shank_num * self.num_contacts_per_shank + num_invalid_dorsal,
                          (shank_num + 1) * self.num_contacts_per_shank - num_invalid_ventral)
                for shank_num, (num_invalid_dorsal, num_invalid_ventral) in enumerate(
                    zip(num_invalid_dorsal_contacts, num_invalid_ventral_contacts))])


"""
# Code for testing LivermoreD2
a = 1*1000
b = .4*1000

below_dorsal_limit_lens = np.asarray([a]*4)
below_ventral_limit_lens = np.asarray([b]*4)
valid_idxs = LivermoreD2().get_valid_idxs(below_dorsal_limit_lens, below_ventral_limit_lens)
print(LivermoreD2().get_num_invalid_dorsal_contacts(a))
print(LivermoreD2().get_num_invalid_ventral_contacts(b))

print(valid_idxs)
print(len(valid_idxs))
"""
