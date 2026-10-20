"""Explicit optional second annotation step, outside the deterministic parser."""
from .core.semantic_v24 import targets, messages, apply_review
from .common import append_json, digest


def review_one(client, problem, response, result, deadline):
    if 'targeted_result' in result or 'parsed' not in result or not targets(result['parsed']):return result
    request = messages(problem, response, result['parsed'])
    review = client.targeted(result.get('request_id',digest({'problem':problem,'response':response})) + ':targeted', request, deadline)
    return finish_review(result,response,request,review)


def finish_review(result,response,request,review):
    updated = dict(result)
    updated['cacheable'] = result.get('cacheable', True) and 'error' not in review
    updated['targeted_request'] = request
    updated['targeted_result'] = review
    updated['initial_parsed'] = result['parsed']
    updated['parsed'] = apply_review(result['parsed'], response, review.get('parsed'), review.get('error'))
    updated['attempts'] = result.get('attempts',0) + review.get('attempts',0)
    updated['usage'] = {key:result.get('usage',{}).get(key,0)+review.get('usage',{}).get(key,0)
                        for key in ('prompt_tokens','completion_tokens','total_tokens')}
    updated['elapsed_seconds'] = result.get('elapsed_seconds',0)+review.get('elapsed_seconds',0)
    return updated


def review_batch(owner, results, unique, deadline, step):
    if not owner.options.get('semantic_check',False):return results
    from .annotation_scheduler import annotate_chains
    contexts={key:(unique[key][0]['prompt'],unique[key][0][unique[key][1]]['response']) for key in results}
    return annotate_chains(owner.judge,{key:None for key in results},contexts,deadline,recovered=results)
