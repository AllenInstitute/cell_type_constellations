def sanitize_taxon_name(name):
    """
    Take a taxon name and replace any problematic
    characters (e.g. '/') with a text representation
    of that character (e.g. '$SLASH$') so that the
    taxon name can be used as a group name in HDF5
    without implying a subgroup that is not there
    """
    new_name = name.replace('/', '$SLASH$')
    return new_name
