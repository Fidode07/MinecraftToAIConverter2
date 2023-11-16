from typing import *

prefix: str = '[MinecraftToAIConverter - Server] '
max_token_length: int = 25
warning_sentences: Dict[str, str] = {
    'no_tag': prefix + 'Skipped intent with index {idx} in dataset: {dataset}. Reason: No tag were found.',
    'duplicated_tag': prefix + 'Skipped intent {tag} in dataset {dataset}, because there is already another '
                               'intent with that tag.',
    'no_patterns': prefix + 'Skipped intent {tag} in dataset {dataset}, because no patterns were given.',
    'empty_pattern': prefix + 'The pattern {pattern} with tag {tag} from the dataset {dataset} were skipped, because'
                              ' it was detected as empty.',
    'max_token_length_exceeded': prefix + 'The sentence {pattern} from the dataset {dataset} with the tag {tag} was '
                                          'skipped, because it exceeds the maximum token length of {max_token_length}. '
                                          'Remove it or increase the max_token_length in utils/data.py'
}
info_sentences: Dict[str, str] = {
    'prepare_data': prefix + 'Prepare training data, before starting training ...',
    'start_training': prefix + 'Start training of classifier model ...'
}
