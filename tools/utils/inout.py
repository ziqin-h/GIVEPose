import json

def save_json(path, content):
  """Saves the provided content to a JSON file.
  :param path: Path to the output JSON file.
  :param content: Dictionary/list to save.
  """
  with open(path, 'w') as f:

    if isinstance(content, dict):
      f.write('{\n')
      content_sorted = sorted(content.items(), key=lambda x: x[0])
      for elem_id, (k, v) in enumerate(content_sorted):
        f.write('  \"{}\": {}'.format(k, json.dumps(v, sort_keys=True)))
        if elem_id != len(content) - 1:
          f.write(',')
        f.write('\n')
      f.write('}')

    elif isinstance(content, list):
      f.write('[\n')
      for elem_id, elem in enumerate(content):
        f.write('  {}'.format(json.dumps(elem, sort_keys=True)))
        if elem_id != len(content) - 1:
          f.write(',')
        f.write('\n')
      f.write(']')

    else:
      json.dump(content, f, sort_keys=True)