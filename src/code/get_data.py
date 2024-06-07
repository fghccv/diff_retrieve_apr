import re, os, tqdm, json
import subprocess
from collections import defaultdict
version = "defects4j-1.2.0"
all_projects = "/home/zhoushiqi/workplace/apr/df4/all_project"
ori_dir = f"/home/zhoushiqi/workplace/apr/df4/{version}/framework/projects"
dest_dir = "/home/zhoushiqi/workplace/apr/data/df4_process_data"
def call_extract(file_path, search_index):
    s = subprocess.run(['java', '-jar', '/home/zhoushiqi/workplace/apr/src/code/extract.jar', file_path, str(search_index)], capture_output=True, text=True)
    result = {}
    result['type'] = re.findall("type:(\w+?)\n", s.stdout)[0]
    if re.findall("range:(.+?)\n", s.stdout) != []:
        result['range'] = re.findall("range:(.+?)\n", s.stdout)[0]
    if re.findall("javadoc:@@@begin@@@((.|\n)*)@@@end@@@", s.stdout) != []:
        result['javadoc'] = re.findall("javadoc:@@@begin@@@((.|\n)*)@@@end@@@", s.stdout)[0][0]
    if re.findall("name:(.+?)\n", s.stdout) != []:
        result['name'] = re.findall("name:(.+?)\n", s.stdout)[0]
    return result
def call_external(file_path, method_name):
    s = subprocess.run(['java', '-jar', '/home/zhoushiqi/workplace/apr/src/code/external.jar', file_path, method_name], capture_output=True, text=True)
    s = s.stdout.split('\n')
    identifiers = []
    if s[0] == 'External References:':
        identifiers = list(set(s[1:]))
        identifiers.remove('')
    else:
        identifiers = []

    return identifiers
def call_extract_func(file_path, method_name):
    s = subprocess.run(['java', '-jar', '/home/zhoushiqi/workplace/apr/src/code/func_extract.jar', file_path, method_name], capture_output=True, text=True)
    s = s.stdout
    result = {}
    if re.findall("Javadoc:((.|\n)*)", s) != []:
        result['javadoc'] = re.findall("Javadoc:((.|\n)*)", s)[0][0]
    if re.findall("Function Body:((.|\n)+)Javadoc:", s) != []:
        result['body'] = re.findall("Function Body:((.|\n)+)Javadoc:", s)[0][0]
    return result
def extract_func(file_path, search_index):
    with open(file_path, encoding="iso-8859-1") as f:
        text = f.read().split('\n')
    result = call_extract(file_path, search_index)
    if result['type'] == 'File':
        start, end = 1, len(text)
        doc = ""
        name = ''
    else:
        start, end = result['range'].split('-')
        doc = result['javadoc']
        name = result['name']
    return text[int(start)-1: int(end)], (int(start)-1, int(end)), doc, result['type'], name
def repair(func_lines, patchs):
    for patch_lines, tags, gap, _, _, _, _ in patchs:
        for i in range(len(func_lines)):
            if func_lines[i:i+gap] == tags:
                p = i
                break
        for patch_line in patch_lines:
            if patch_line.startswith("@@"): continue
            if patch_line.startswith("-"):
                func_lines.insert(p, patch_line[1:])
            elif patch_line.startswith("+") :
                del func_lines[p]
                continue
            p += 1
    return '\n'.join(func_lines)
def add_erro(func_lines, patchs):
    for patch_lines, tags, gap, _, _, _ in patchs:
        for i in range(len(func_lines)):
            if func_lines[i:i+gap] == tags:
                p = i
                break
        for patch_line in patch_lines:
            if patch_line.startswith("@@"): continue
            if patch_line.startswith("+"):
                func_lines[p] += '//There might be a bug near this line of code (including this line).'
            elif patch_line.startswith("-"):
                continue
            p += 1
    return '\n'.join(func_lines)
def get_info(ori_dir, df4_version="defects4j-1.2.0"):
    infos = []
    for project in os.listdir(ori_dir):
        if project == 'lib':continue
        path = ori_dir + '/{}'.format(project)
        if not os.path.isdir(path):
            continue
        print(project)
        #提取patch和相关test
        tests = sorted([path + '/trigger_tests/' + f for f in os.listdir(path + '/trigger_tests')])
        patches = sorted([path + '/patches/' + f for f in os.listdir(path + '/patches') if 'src' in f])
        assert len(tests) == len(patches)
        for test, patch in tqdm.tqdm(zip(tests, patches)):
            info = {}
            info['version'] = df4_version
            info['project'] = project
            bug_id = int(test.split('/')[-1])
            # if bug_id == 8 and project=='Math':
            #     a = 0
            # else:
            #     continue
            info['bug_id'] = bug_id
            prefix_project = f"{all_projects}/{df4_version}/{project}/{str(bug_id)}"
            if not os.path.exists(prefix_project):
                continue
            with open(prefix_project + '/defects4j.build.properties') as f:
                for line in f:
                    if line.startswith("#"): continue
                    key, value = line.split('=')
                    if "d4j.dir.src.classes" in key:
                        src_relative = value.strip()
                    if "d4j.dir.src.tests" in key:
                        test_relative = value.strip()
            temp_dest_dir = dest_dir + f'/{df4_version}/{project}/{str(bug_id)}/'
            try:
                with open(test) as f1, open(patch) as f2:
                    test_text, patch_text = f1.read(), f2.read()
            except:
                with open(test) as f1, open(patch, encoding="iso-8859-1") as f2:
                    test_text, patch_text = f1.read(), f2.read()
            #提取test
            assert test_text.startswith('---')
            erro_class, test_fun = test_text.split("\n")[0].split()[-1].split('::')
            erro_track = test_text.split("\n")[1:]
            test_path = subprocess.run(["find", prefix_project + f'/{test_relative}', "-type","f", "-path", "*/"+erro_class.replace('.','/')+'.java'], capture_output=True, text=True).stdout.strip()
            info['Test_file_path'] = test_path
            new_test_path = temp_dest_dir + 'test.java'
            new_trigger_path = temp_dest_dir + 'trigger.java'
            new_erro_path = temp_dest_dir + 'erro.java'
            new_all_testcase_path = temp_dest_dir + 'all_test_case.java'
            subprocess.run(["mkdir", "-p", new_test_path.strip("test.java")])
            subprocess.run(["cp", test_path, new_test_path])
            subprocess.run(["cp", test, new_erro_path])
            info['local_dir_path'] = new_test_path.strip("test.java")
            for erro in erro_track:
                if erro_class + '.' + test_fun in erro:
                    erro_line_index = int(re.findall('\(\w+\.java:(\d+)\)', erro)[0])
                    with open(new_trigger_path, 'w') as f, open(test_path, encoding="iso-8859-1") as f2:
                        lines = f2.read().split('\n')
                        f.write(str(erro_line_index) + ' ' + lines[erro_line_index-1].strip())
                        info['trigger_line'] = lines[erro_line_index-2:erro_line_index]#erro_line_index-1是错误行
                    break
            with open(new_all_testcase_path, 'w') as f:
                try:
                    test_func = '\n'.join(extract_func(test_path, erro_line_index)[0])
                    if 'trigger_line' in info:
                        ii = test_func.index('\n'.join(info['trigger_line'])) + len('\n'.join(info['trigger_line'][:2]))
                        test_func = test_func[:ii] + "//The program encountered an error here" + test_func[ii:]
                    f.write(test_func)
                    info['test_func'] = test_func
                except:
                    info['extract_test_erro'] = True
                    print('erro extact test:%s %s'%(project, bug_id))
            #提取原始函数
            patch_lines = patch_text.split('\n')
            diff_indexs = [i for i in range(len(patch_lines)) if "diff --git" in patch_lines[i]]
            info['erro_repairs'] = []
            for i in range(len(diff_indexs)):
                one_erro = {}
                if i!=len(diff_indexs)-1:
                    s = '\n'.join(patch_lines[diff_indexs[i]:diff_indexs[i+1]])
                else:
                    s = '\n'.join(patch_lines[diff_indexs[i]:])
                file_name = re.findall("(((src)|(source)).*?\.((java)|(txt)))", s)[0][0]
                src_path = subprocess.run(["find", prefix_project + f"/{src_relative}", "-type","f", "-path", "*/"+file_name], capture_output=True, text=True).stdout.strip()
                one_erro['src_path'] = src_path
                new_src_path = temp_dest_dir + f'src{i}.java'
                new_diff_path = temp_dest_dir + 'diff'+str(i)+'.patch'
                err_func_path = temp_dest_dir + 'erro_func'+str(i)+'.java'
                with open(new_diff_path, 'w') as f:
                    f.write(s)
                if src_path != "":
                    subprocess.run(["cp", src_path, new_src_path])
                else:
                    info['extract_src_erro'] = True
                    print("no src file!")
                    continue
                with open(err_func_path, 'w') as f:
                    try:
                        diff_lines = s.strip('\n').split('\n')
                        diff_is = [i for i in range(len(diff_lines)) if re.search("@@\s[+|-]\d+,\d+\s[+|-]*(\d+),\d+\s@@", diff_lines[i]) != None]
                        diff_is.append(len(diff_lines))
                        line_indexs = []
                        diff_ls = []
                        one_erro['fixs'] = []
                        one_erro['src_code'] = []
                        one_erro['patchs'] = []
                        for i in range(len(diff_is) - 1):
                            di = diff_is[i]
                            r = re.findall("@@\s[+|-]\d+,\d+\s[+|-]*(\d+),(\d+)\s@@", diff_lines[di])
                            p, q = r[0]
                            kk = di+1
                            while not (diff_lines[kk].startswith('-') or diff_lines[kk].startswith('+')):
                                kk += 1
                            start = int(p) + (kk-di-1)
                            cxs = []
                            kke = kk
                            while kke < diff_is[i+1]:
                                if diff_lines[kke].startswith('-'):
                                    cxs.append(0)
                                elif diff_lines[kke].startswith('+'):
                                    cxs.append(1)
                                else:
                                    cxs.append(-1)
                                kke += 1
                            kke -= 1
                            while cxs[-1] == -1:
                                cxs = cxs[:-1]
                                kke -= 1
                            gap = cxs.count(-1)+cxs.count(1)
                            if gap == 0:
                                gap = 1
                                line_indexs.append([start-1, gap])
                                diff_ls.append(diff_lines[kk-1:kke+1])        
                            else:
                                line_indexs.append([start, gap])
                                diff_ls.append(diff_lines[kk:kke+1])                                     
                                
                        src_lines = open(src_path, encoding="iso-8859-1").read().split('\n')
                        func2diffs = defaultdict(list)
                        for (line_index, gap), diff_l in zip(line_indexs, diff_ls):
                            tags = src_lines[line_index-1:line_index+gap-1]
                            starts, ends = [], []
                            _, rg1, doc1, ty1, name1 = extract_func(src_path, line_index)
                            _, rg2, doc2, ty2, name2 = extract_func(src_path, line_index+gap-1)
                            identifiers1 = call_external(src_path, name1)
                            identifiers2 = call_external(src_path, name2)
                            for start, end in [rg1, rg2]:
                                starts.append(start)
                                ends.append(end)
                            if 'File' in [ty1, ty2]:
                                ty = 'File'
                            elif 'class' in [ty1, ty2]:
                                ty = 'class'
                            else:
                                ty = 'Method'
                            identifiers = list(set(identifiers1+identifiers2))
                            if name1 in identifiers:
                                identifiers.remove(name1)
                            if name2 in identifiers:
                                identifiers.remove(name2)
                            func2diffs[(min(starts), max(ends))].append([diff_l, tags, gap, [rg1, rg2], ty, list(set([doc1, doc2])), identifiers])
                        one_erro['if_one_function'] = []
                        one_erro['docs'] = []
                        one_erro['externals'] = []
                        one_erro['relevent_method'] = []
                        for func_i in func2diffs:
                            all_docs = []
                            all_ty = []
                            all_rgs = []
                            all_externals = []
                            for x in func2diffs[func_i]:
                                all_externals += x[-1]
                                all_docs += x[-2]
                                all_ty.append(x[-3])
                                all_rgs += x[-4]
                            one_relevent_method = []
                            for ide in all_externals:
                                cf = call_extract_func(src_path, ide)
                                if cf != {}:
                                    one_relevent_method.append(cf)
                            one_erro['relevent_method'].append(one_relevent_method)
                            one_erro['externals'].append(all_externals)
                            one_erro['docs'].append(all_docs)
                            one_erro['if_one_function'].append(all(ty=='Method' for ty in all_ty) and len(set(all_rgs))==1)
                            final_func = '\n'.join(src_lines[func_i[0]:func_i[1]])
                            # final_func = add_erro(src_lines[func_i[0]:func_i[1]], func2diffs[func_i])
                            f.write(final_func)
                            one_erro['src_code'].append(final_func)
                            f.write("\n################ repair func ##################\n")
                            final_repair = repair(src_lines[func_i[0]:func_i[1]], func2diffs[func_i])
                            one_erro['patchs'].append(func2diffs[func_i])
                            f.write(final_repair)
                            f.write("\n################ next func ##################\n")
                            one_erro['fixs'].append(final_repair)
                    except:
                        info['extract_src_erro'] = True
                        print('erro extact func:%s %s'%(project, bug_id))
                info['erro_repairs'].append(one_erro)  
            infos.append(info)
    with open(f"/home/zhoushiqi/workplace/apr/data/df4_process_data/all_info_{df4_version}_external.jsonl", 'w') as f:
        for info in infos:
            f.write(json.dumps(info) + '\n')

get_info(ori_dir, version)