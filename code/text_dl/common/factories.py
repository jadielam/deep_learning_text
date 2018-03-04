def generic_factory(type_factory_d, factory_type_s):
    '''
    Meta factory function

    Arguments:
        - type_factory_d (Dict[string -> callable])
        - factory_type_s (string)
    
    Returns:
        - factory (callable): the function to be used as factory
    '''
    def factory(conf):
        '''
        Given a configuration, returns a created object passing conf as
        parameter

        Arguments:
            - conf (dict): Configuration with parameters on it
        
        Returns:
            - object: Created object
        '''
        ftype = conf['type']
        params = conf.get("params", {})
        try:
            return type_factory_d[ftype](**params)
        except KeyError:
            raise ValueError("Incorrect {} type: {}".format(factory_type_s, ftype))

    return factory
