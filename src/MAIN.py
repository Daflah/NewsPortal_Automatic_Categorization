'''
=============================================================================================================================
Author  : Daflah Tsany Gusra
Build   : 11 Oktober 2025
Purpose : Main pipeline untuk menjalankan seluruh proses klasifikasi berita
=============================================================================================================================
'''

import os
import sys
import time
from datetime import datetime
import threading
import time

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
import CONFIG

# Import semua modul
sys.path.append(os.path.dirname(__file__))
from CNP_PREPROCESSING import test_preprocessing, get_dataframe_with_processed_text, save_processed_data
from CNP_METODE_BOW import test_bow
from CNP_METODE_LDA import test_lda
from CNP_METODE_RF import test_random_forest

class ProgressTracker:
    """Class untuk menampilkan progress bar dan estimasi waktu"""
    
    @staticmethod
    def animated_loading(text):
        """Animasi loading sederhana"""
        for i in range(3):
            sys.stdout.write(f'\r{text}{"." * (i + 1)}   ')
            sys.stdout.flush()
            time.sleep(0.5)
        sys.stdout.write('\r' + ' ' * (len(text) + 5) + '\r')
    
    @staticmethod
    def progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
        """Progress bar tradisional"""
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
        sys.stdout.flush()
        if iteration == total:
            print()

class NewsPortalPipeline:
    def __init__(self):
        self.start_time = None
        self.results = {}
        self.step_times = {}
        
    def start_pipeline(self):
        """Memulai seluruh pipeline"""
        self.start_time = time.time()
        print("=" * 80)
        print("NEWS PORTAL CLASSIFICATION PIPELINE")
        print("=" * 80)
        print(f"Mode: {'TESTING' if CONFIG.TESTING_MODE else 'PRODUCTION'}")
        print(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
    def end_pipeline(self):
        """Mengakhiri pipeline dan menampilkan summary"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED - SUMMARY")
        print("=" * 80)
        print(f"Total waktu eksekusi: {self._format_time(duration)}")
        
        if self.results:
            print("\n📊 Detail setiap tahap:")
            for step, result in self.results.items():
                step_time = self.step_times.get(step, 0)
                if result.get('success'):
                    print(f"  ✅ {step}: {result.get('message', 'Berhasil')} ({self._format_time(step_time)})")
                else:
                    print(f"  ❌ {step}: {result.get('message', 'Gagal')} ({self._format_time(step_time)})")
        
        print("=" * 80)
    
    def _format_time(self, seconds):
        """Format waktu menjadi string yang mudah dibaca"""
        if seconds < 60:
            return f"{seconds:.1f} detik"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} menit"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} jam"
    
    def _estimate_step_time(self, step_name, dataset_size):
        """Estimasi waktu untuk setiap step berdasarkan dataset size"""
        base_times = {
            'Preprocessing': 0.001,  # detik per dokumen
            'Bag-of-Words': 0.0005,  # detik per dokumen
            'LDA': 0.002,            # detik per dokumen
            'Random Forest': 0.0003   # detik per dokumen
        }
        
        if CONFIG.TESTING_MODE:
            multiplier = 0.1  # Lebih cepat di testing mode
        else:
            multiplier = 1.0
        
        estimated_time = base_times.get(step_name, 0.001) * dataset_size * multiplier
        return estimated_time
    
    def _show_progress_animation(self, step_name, estimated_time):
        """Menampilkan progress animation dengan estimasi waktu"""
        print(f"\n🔄 {step_name}...")
        print(f"   ⏱️  Estimasi: {self._format_time(estimated_time)}")
        
        # Start animation in separate thread
        def animate():
            dots = 0
            start_time = time.time()
            while not getattr(threading.current_thread(), "stop", False):
                elapsed = time.time() - start_time
                dots = (dots + 1) % 4
                sys.stdout.write(f'\r   📊 Sedang memproses{"." * dots} ({self._format_time(elapsed)})')
                sys.stdout.flush()
                time.sleep(0.5)
        
        animation_thread = threading.Thread(target=animate)
        animation_thread.stop = False
        animation_thread.start()
        return animation_thread
    
    def _stop_progress_animation(self, animation_thread):
        """Menghentikan progress animation"""
        animation_thread.stop = True
        animation_thread.join()
        sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear line
    
    def run_preprocessing(self):
        """Menjalankan preprocessing"""
        print("\n" + "=" * 50)
        print("STEP 1: TEXT PREPROCESSING")
        print("=" * 50)
        
        # Estimasi waktu berdasarkan dataset
        dataset_size = 25000 if not CONFIG.TESTING_MODE else CONFIG.TEST_SAMPLE_SIZE
        estimated_time = self._estimate_step_time('Preprocessing', dataset_size)
        
        animation_thread = self._show_progress_animation('Preprocessing', estimated_time)
        step_start = time.time()
        
        try:
            # Jalankan preprocessing
            test_preprocessing()
            
            # Verifikasi file hasil
            if CONFIG.TESTING_MODE:
                output_file = os.path.join(
                    CONFIG.PROCESSED_DATA_PATH, 
                    f"processed_data_test_{CONFIG.TEST_FILE.split('.')[0]}.csv"
                )
            else:
                output_file = os.path.join(CONFIG.PROCESSED_DATA_PATH, "processed_data_full.csv")
            
            self._stop_progress_animation(animation_thread)
            step_duration = time.time() - step_start
            
            if os.path.exists(output_file):
                df = get_dataframe_with_processed_text()
                self.results['preprocessing'] = {
                    'success': True,
                    'message': f"{len(df)} dokumen berhasil diproses",
                    'file': output_file
                }
                self.step_times['preprocessing'] = step_duration
                print(f"✅ Preprocessing selesai! ({self._format_time(step_duration)})")
                return True
            else:
                self.results['preprocessing'] = {
                    'success': False,
                    'message': 'File output preprocessing tidak ditemukan'
                }
                self.step_times['preprocessing'] = step_duration
                print(f"❌ Preprocessing gagal ({self._format_time(step_duration)})")
                return False
                
        except Exception as e:
            self._stop_progress_animation(animation_thread)
            step_duration = time.time() - step_start
            print(f"❌ ERROR dalam preprocessing: {e}")
            self.results['preprocessing'] = {
                'success': False,
                'message': f'Error: {str(e)}'
            }
            self.step_times['preprocessing'] = step_duration
            return False
    
    def run_bow(self):
        """Menjalankan Bag-of-Words"""
        print("\n" + "=" * 50)
        print("STEP 2: BAG-OF-WORDS")
        print("=" * 50)
        
        # Estimasi waktu
        dataset_size = 25000 if not CONFIG.TESTING_MODE else CONFIG.TEST_SAMPLE_SIZE
        estimated_time = self._estimate_step_time('Bag-of-Words', dataset_size)
        
        animation_thread = self._show_progress_animation('Bag-of-Words', estimated_time)
        step_start = time.time()
        
        try:
            # Jalankan BOW
            bow_matrix, vocabulary, df = test_bow()
            
            self._stop_progress_animation(animation_thread)
            step_duration = time.time() - step_start
            
            if bow_matrix is not None:
                # Verifikasi file hasil
                bow_results_file = os.path.join(CONFIG.BOW_OUTPUT_DIR, "bow_results.csv")
                bow_lda_file = os.path.join(CONFIG.BOW_OUTPUT_DIR, "bow_for_lda.csv")
                
                if os.path.exists(bow_results_file) and os.path.exists(bow_lda_file):
                    self.results['bow'] = {
                        'success': True,
                        'message': f"BOW matrix {bow_matrix.shape} berhasil dibuat",
                        'files': [bow_results_file, bow_lda_file]
                    }
                    self.step_times['bow'] = step_duration
                    print(f"✅ Bag-of-Words selesai! ({self._format_time(step_duration)})")
                    return True
                else:
                    self.results['bow'] = {
                        'success': False,
                        'message': 'File output BOW tidak ditemukan'
                    }
                    self.step_times['bow'] = step_duration
                    print(f"❌ Bag-of-Words gagal ({self._format_time(step_duration)})")
                    return False
            else:
                self.results['bow'] = {
                    'success': False,
                    'message': 'Proses BOW gagal'
                }
                self.step_times['bow'] = step_duration
                print(f"❌ Bag-of-Words gagal ({self._format_time(step_duration)})")
                return False
                
        except Exception as e:
            self._stop_progress_animation(animation_thread)
            step_duration = time.time() - step_start
            print(f"❌ ERROR dalam BOW: {e}")
            self.results['bow'] = {
                'success': False,
                'message': f'Error: {str(e)}'
            }
            self.step_times['bow'] = step_duration
            return False
    
    def run_lda(self):
        """Menjalankan LDA"""
        print("\n" + "=" * 50)
        print("STEP 3: LATENT DIRICHLET ALLOCATION (LDA)")
        print("=" * 50)
        
        # Estimasi waktu (LDA biasanya paling lama)
        dataset_size = 25000 if not CONFIG.TESTING_MODE else CONFIG.TEST_SAMPLE_SIZE
        estimated_time = self._estimate_step_time('LDA', dataset_size)
        
        animation_thread = self._show_progress_animation('LDA', estimated_time)
        step_start = time.time()
        
        try:
            # Jalankan LDA
            lda_model, doc_topic_dist, topic_word_dist = test_lda()
            
            self._stop_progress_animation(animation_thread)
            step_duration = time.time() - step_start
            
            if lda_model is not None:
                # Verifikasi file hasil
                lda_results_file = os.path.join(CONFIG.LDA_OUTPUT_DIR, "lda_results.csv")
                topic_words_file = os.path.join(CONFIG.LDA_OUTPUT_DIR, "topic_words.csv")
                
                if os.path.exists(lda_results_file) and os.path.exists(topic_words_file):
                    self.results['lda'] = {
                        'success': True,
                        'message': f"LDA dengan {CONFIG.N_TOPICS} topik berhasil",
                        'files': [lda_results_file, topic_words_file]
                    }
                    self.step_times['lda'] = step_duration
                    print(f"✅ LDA selesai! ({self._format_time(step_duration)})")
                    return True
                else:
                    self.results['lda'] = {
                        'success': False,
                        'message': 'File output LDA tidak ditemukan'
                    }
                    self.step_times['lda'] = step_duration
                    print(f"❌ LDA gagal ({self._format_time(step_duration)})")
                    return False
            else:
                self.results['lda'] = {
                    'success': False,
                    'message': 'Proses LDA gagal'
                }
                self.step_times['lda'] = step_duration
                print(f"❌ LDA gagal ({self._format_time(step_duration)})")
                return False
                
        except Exception as e:
            self._stop_progress_animation(animation_thread)
            step_duration = time.time() - step_start
            print(f"❌ ERROR dalam LDA: {e}")
            self.results['lda'] = {
                'success': False,
                'message': f'Error: {str(e)}'
            }
            self.step_times['lda'] = step_duration
            return False
    
    def run_random_forest(self):
        """Menjalankan Random Forest"""
        print("\n" + "=" * 50)
        print("STEP 4: RANDOM FOREST CLASSIFICATION")
        print("=" * 50)
        
        # Estimasi waktu
        dataset_size = 25000 if not CONFIG.TESTING_MODE else CONFIG.TEST_SAMPLE_SIZE
        estimated_time = self._estimate_step_time('Random Forest', dataset_size)
        
        animation_thread = self._show_progress_animation('Random Forest', estimated_time)
        step_start = time.time()
        
        try:
            # Jalankan Random Forest
            rf_model, accuracy = test_random_forest()
            
            self._stop_progress_animation(animation_thread)
            step_duration = time.time() - step_start
            
            if rf_model is not None:
                # Verifikasi file hasil
                rf_results_file = os.path.join(CONFIG.MODEL_RESULTS_DIR, "rf_results.csv")
                model_metrics_file = os.path.join(CONFIG.MODEL_RESULTS_DIR, "model_metrics.json")
                
                if os.path.exists(rf_results_file):
                    self.results['random_forest'] = {
                        'success': True,
                        'message': f"Random Forest accuracy: {accuracy:.4f}",
                        'files': [rf_results_file, model_metrics_file]
                    }
                    self.step_times['random_forest'] = step_duration
                    print(f"✅ Random Forest selesai! ({self._format_time(step_duration)})")
                    return True
                else:
                    self.results['random_forest'] = {
                        'success': False,
                        'message': 'File output Random Forest tidak ditemukan'
                    }
                    self.step_times['random_forest'] = step_duration
                    print(f"❌ Random Forest gagal ({self._format_time(step_duration)})")
                    return False
            else:
                self.results['random_forest'] = {
                    'success': False,
                    'message': 'Proses Random Forest gagal'
                }
                self.step_times['random_forest'] = step_duration
                print(f"❌ Random Forest gagal ({self._format_time(step_duration)})")
                return False
                
        except Exception as e:
            self._stop_progress_animation(animation_thread)
            step_duration = time.time() - step_start
            print(f"❌ ERROR dalam Random Forest: {e}")
            self.results['random_forest'] = {
                'success': False,
                'message': f'Error: {str(e)}'
            }
            self.step_times['random_forest'] = step_duration
            return False
    
    def run_full_pipeline(self):
        """Menjalankan seluruh pipeline dari awal sampai akhir"""
        self.start_pipeline()
        
        # Tampilkan estimasi total
        total_dataset_size = 25000 if not CONFIG.TESTING_MODE else CONFIG.TEST_SAMPLE_SIZE
        total_estimated_time = sum([
            self._estimate_step_time('Preprocessing', total_dataset_size),
            self._estimate_step_time('Bag-of-Words', total_dataset_size),
            self._estimate_step_time('LDA', total_dataset_size),
            self._estimate_step_time('Random Forest', total_dataset_size)
        ])
        
        print(f"📊 Estimasi total waktu: {self._format_time(total_estimated_time)}")
        print(f"💾 Mode: {'TESTING' if CONFIG.TESTING_MODE else 'PRODUCTION'}")
        if CONFIG.TESTING_MODE:
            print(f"🔬 Sample size: {CONFIG.TEST_SAMPLE_SIZE} dokumen")
        else:
            print(f"📁 Full dataset: ~25,000 dokumen")
        print()
        
        steps = [
            ('Preprocessing', self.run_preprocessing),
            ('Bag-of-Words', self.run_bow),
            ('LDA', self.run_lda),
            ('Random Forest', self.run_random_forest)
        ]
        
        # Jalankan setiap step
        successful_steps = 0
        for step_name, step_function in steps:
            print(f"🚀 Memulai {step_name}...")
            success = step_function()
            
            if success:
                successful_steps += 1
                print(f"✅ {step_name} berhasil!")
            else:
                print(f"❌ Pipeline dihentikan karena {step_name} gagal")
                break
            
            print("-" * 50)
        
        self.end_pipeline()
        
        return successful_steps == len(steps)

def main():
    """Main function"""
    pipeline = NewsPortalPipeline()
    
    try:
        success = pipeline.run_full_pipeline()
        
        if success:
            print("\n🎉 SELAMAT! Seluruh pipeline berhasil dijalankan!")
            print("📁 File hasil dapat ditemukan di:")
            print(f"   - Preprocessing: {CONFIG.PROCESSED_DATA_PATH}")
            print(f"   - BoW: {CONFIG.BOW_OUTPUT_DIR}")
            print(f"   - LDA: {CONFIG.LDA_OUTPUT_DIR}")
            print(f"   - Random Forest: {CONFIG.MODEL_RESULTS_DIR}")
        else:
            print("\n⚠️ Pipeline memiliki beberapa kegagalan. Periksa log di atas.")
            
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline dihentikan oleh user")
    except Exception as e:
        print(f"\n💥 ERROR tidak terduga: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()