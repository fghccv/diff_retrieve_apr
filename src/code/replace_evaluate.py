import utils,tqdm, json
import subprocess,os
import signal
from threading import Thread,Lock
from concurrent.futures import ThreadPoolExecutor
import argparse, os, utils
n = 0
nowLock = Lock()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', type=str, help="")
    parser.add_argument('--infos_path', type=str, help="")
    parser.add_argument('--process_path', type=str, help="")
    parser.add_argument('--result_path', type=str, help="")
    parser.add_argument('--temp_name', type=str, help="", default="temp")
    args = parser.parse_args()
    
def my_function(test_class, path, if_all=False):
    try:
        # 设置subprocess.run()的超时时间为600秒
        if if_all:
            return subprocess.run(['defects4j', 'test', '-r', '-w', path], capture_output=True, text=True, timeout=240)
        else:
            return subprocess.run(['defects4j', 'test', '-t', test_class, '-w', path], capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        raise TimeoutError("Execution timed out")
version = args.version
if version == '1.7':
    defects_dir = "/home/zhoushiqi/workplace/apr/df4/defects4j-1.2.0"
else:
    defects_dir = "/home/zhoushiqi/workplace/apr/df4/defects4j-2.0.1"
# 获取当前的PATH环境变量值
current_path = os.environ.get('PATH', '')

# 要添加的目录路径
new_path = f'{defects_dir}/framework/bin:/home/zhoushiqi/lib/jvm/jdk{version}/bin'

# 将新目录路径添加到PATH环境变量中
new_path_value = f'{new_path}:{current_path}' if current_path else new_path
os.environ['PATH'] = new_path_value
os.system("java -version")
os.system("defects4j info -p Lang")
infos_path = args.infos_path
process_path = args.process_path
result_path = args.result_path
temp_path = '/home/zhoushiqi/workplace/apr/data/' + args.temp_name
infos = utils.read_jsonl(infos_path)
process = json.load(open(process_path))
if os.path.exists(result_path):
    results = utils.read_jsonl(result_path)
else:
    results = []
exsists = [f"{result['project']}_{result['bug_id']}" for result in results]
# results = []
# exsists = []
def func(t, progress_bar, thread_id):
    global n, nowLock
    thread_temp_path = f"{temp_path}/{thread_id}"
    while n < t:

        nowLock.acquire()

        info = infos[n]
        try:
            process_data = process[f"{info['project']}_{info['bug_id']}"]
        except Exception:
            n += 1
            nowLock.release()
            progress_bar.update(1)
            continue
        n += 1
        nowLock.release()


        if f"{info['project']}_{info['bug_id']}" not in exsists:

            result = []
            src_path = info['erro_repairs'][0]['src_path']
            try:
                ori_java_text = open(src_path).read()
            except:
                ori_java_text = open(src_path, encoding="iso-8859-1").read()
            ori_code = info['erro_repairs'][0]['src_code'][0]
            version = info['version']
            project = info['project']
            id = info['bug_id']
            # os.system(f"cp {info['local_dir_path']}/src0.java {info['local_dir_path']}/fix0.java")
            # subprocess.run(['python', '/home/zhoushiqi/anaconda3/envs/apr/lib/python3.10/site-packages/dos2unix.py', f"{info['local_dir_path']}/fix0.java", f"{info['local_dir_path']}/fix0.java"], capture_output=True, text=True)
            # subprocess.run(['python', '/home/zhoushiqi/anaconda3/envs/apr/lib/python3.10/site-packages/dos2unix.py', f"{info['local_dir_path']}/diff0.patch", f"{info['local_dir_path']}/diff0.patch"], capture_output=True, text=True)
            # x = subprocess.run(['patch', '-R', f"{info['local_dir_path']}/fix0.java", f"{info['local_dir_path']}/diff0.patch"], capture_output=True, text=True)
            test_class = open(f"{info['local_dir_path']}/erro.java").read().split('\n')[0].replace('--- ','')
            ori_dir = f'/home/zhoushiqi/workplace/apr/df4/all_project/{version}/{project}/{id}/'
            new_src_path = thread_temp_path + src_path.replace(ori_dir, '/')
            if os.path.exists(thread_temp_path):
                os.system(f"rm -r -f {thread_temp_path}")
            os.system(f"mkdir -p {thread_temp_path}")
            os.system(f"cp -r {ori_dir}/. {thread_temp_path}")
            repair_codes = process_data
            for repair_code in repair_codes:
                if repair_code == '':
                    result.append('Generate Failing')
                    continue
                if repair_code.strip() == ori_code.strip():
                    result.append("Repetite Failed")
                    continue
                repai_java_text = ori_java_text.replace(ori_code, repair_code)
                # repai_java_text = open(f"{info['local_dir_path']}/fix0.java", encoding="iso-8859-1").read()
                try:
                    with open(new_src_path, 'w', encoding="utf-8") as f:
                        f.write(repai_java_text)
                except:
                    with open(new_src_path, 'w', encoding="iso-8859-1") as f:
                        f.write(repai_java_text)
                c = subprocess.run(['defects4j', 'compile', '-w', thread_temp_path], capture_output=True, text=True)
                if 'OK' not in c.stderr:
                    result.append('Compile Failing')
                    continue
                try:
                    s = my_function(test_class, thread_temp_path)
                    if s.stdout == "Failing tests: 0\n":
                        s = my_function(test_class, thread_temp_path, True)
                        result.append("other test "+s.stdout)
                        # pass
                    else:
                        result.append("trigger test"+s.stdout)
                        pass
                    if 'other test Failing tests: 0\n' in result:
                        break
                except:
                    result.append("Time out")
            print(result)
            results.append({'project':info['project'], 'bug_id':info['bug_id'],'result':result})
            utils.write_jsonl(result_path, results)

        progress_bar.update(1)

def main(num_thread):
    t = len(infos)
    progress_bar = tqdm.tqdm(total=t)
    threadlist = []
    for i in range(num_thread):
        thread = Thread(target=func,
                        args=(t, progress_bar, i)
                        )
        thread.start()
        # 把线程对象都存储到 threadlist中
        threadlist.append(thread)
    for thread in threadlist:
        thread.join()


main(128)