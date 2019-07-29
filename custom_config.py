def get_attachment_path(problem_number):
    '''
    Given a problem number, return the path of the attachment file.
    '''
    import os
    # hard-coded configs
    attachment_dir = 'attachments'
    attachment_filenames = {
                            22: 'p022_names.txt',
                            42: 'p042_words.txt',
                            54: 'p054_poker.txt',
                            59: 'p059_cipher.txt',
                            }
    attachment_path = os.path.join(attachment_dir, attachment_filenames[problem_number])
    return attachment_path
