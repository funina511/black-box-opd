"""Bind metadata to the selected original teacher candidate, never text.index()."""
import hashlib


VERSION = 'teacher.selection.v2'


def bind_selected(answers, sampling, selected_text, selected_index=None):
    matches = [i for i, text in enumerate(answers) if text == selected_text]
    if selected_index is None:
        if len(matches) != 1:
            raise ValueError('teacher candidate cannot be uniquely identified; explicit selected index required')
        selected_index = matches[0]
    if type(selected_index) is not int or selected_index not in matches:
        raise ValueError('selected teacher index/text mismatch')
    def field(name, default):
        values = sampling.get(name) or []
        if len(values) != len(answers):
            if len(answers) != 1:raise ValueError('multi-candidate sampling metadata is incomplete')
            return default
        return values[selected_index]
    finish = field('finish_reasons', None)
    truncated = bool(field('truncated', finish == 'length')) or finish == 'length'
    return {'teacher_metadata_version': VERSION, 'selected_teacher_index': selected_index,
        'selected_teacher_text_sha256': hashlib.sha256(selected_text.encode()).hexdigest(),
        'source_finish_reason': finish, 'source_truncated': truncated}
