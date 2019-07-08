def get_attachment_path(problem_number):
    '''
    Given a problem number, return the path of the attachment file.
    '''
    # hard-coded configs
    attachment_dir = 'attachments'
    attachment_filenames = {22: 'p022_names.txt'}

    import os
    attachment_path = os.path.join(attachment_dir, attachment_filenames[problem_number])
    return attachment_path
